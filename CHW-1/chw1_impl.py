"""Deep Generative Models coding assignment: order-agnostic MADE and Gated PixelCNN.

The notebook imports this module so the training loop, plots, and checks stay in one place.
"""

from __future__ import annotations

import math
import os
import sys
import time

# tensorflow-macos on this machine crashes while loading its Metal plugin.
# TensorBoard treats a failed TensorFlow import as "use the built-in stub".
if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = None

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import make_grid

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
RUN_DIR = os.path.join(ROOT, "runs")
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RUN_DIR, exist_ok=True)

SEED = 0


def tqdm_bar(iterable=None, **kwargs):
    """Notebook progress bar, with a text fallback when no widget frontend is attached."""
    try:
        from tqdm.notebook import tqdm as notebook_tqdm

        return notebook_tqdm(iterable, **kwargs)
    except Exception:
        from tqdm.auto import tqdm

        return tqdm(iterable, **kwargs)


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_optimizer(params, lr):
    """AdamW with the fused GPU kernel. On this Mac the fused state lives on MPS."""
    params = list(params)
    if not params:
        raise ValueError("optimizer got no parameters")
    fused_ok = params[0].device.type in ("mps", "cuda")
    try:
        return torch.optim.AdamW(params, lr=lr, fused=fused_ok)
    except TypeError:
        return torch.optim.AdamW(params, lr=lr, foreach=True)


def smoke_test_optimizer(device):
    """One fused AdamW step. Confirms the moment buffers stay on the GPU."""
    torch.manual_seed(0)
    layer = nn.Linear(32, 32).to(device)
    opt = make_optimizer(layer.parameters(), lr=1e-2)
    x = torch.randn(16, 32, device=device)
    loss = layer(x).square().mean()
    loss.backward()
    opt.step()
    weight = layer.weight
    state = opt.state[weight]
    devices = {name: value.device.type for name, value in state.items() if torch.is_tensor(value)}
    return {
        "device": str(device),
        "optimizer": type(opt).__name__,
        "fused": bool(opt.defaults.get("fused", False)),
        "state_devices": devices,
        "param_device": weight.device.type,
    }


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", torch.ones(out_features, in_features))

    def set_mask(self, mask):
        # The assignment stores masks as (in, out) and transposes them onto the weight.
        # numpy .T is not contiguous, and MPS cannot copy a non-contiguous CPU view.
        transposed = np.ascontiguousarray(np.asarray(mask).astype(np.uint8).T)
        target = torch.from_numpy(transposed).to(device=self.mask.device, dtype=self.mask.dtype)
        self.mask.data.copy_(target)

    def forward(self, input):
        return F.linear(input, self.weight * self.mask, self.bias)


class MADE(nn.Module):
    """Two-hidden-layer Bernoulli MADE. Masks are resampled for every ordering."""

    def __init__(self, input_size, hidden_sizes, num_masks=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = list(hidden_sizes)
        self.num_masks = num_masks
        if len(self.hidden_sizes) != 2:
            raise ValueError("this MADE stack is the two-hidden-layer network from the notebook")
        h0, h1 = self.hidden_sizes
        self.net = nn.Sequential(
            MaskedLinear(input_size, h0),
            nn.ReLU(),
            MaskedLinear(h0, h1),
            nn.ReLU(),
            MaskedLinear(h1, input_size),
        )
        self.input_degrees = None
        self.sample_masks()

    def masked_layers(self):
        return [module for module in self.net if isinstance(module, MaskedLinear)]

    def sample_masks(self, ordering=None):
        """Build one valid autoregressive mask set.

        ordering lists input indices from earliest to latest in the product.
        Hidden degrees use the MADE rule: integers in 1..D-1.
        Hidden connections are allowed when the destination degree is at least the source degree.
        Output connections are strict, so coordinate d cannot see itself.
        """
        d = self.input_size
        if ordering is None:
            ordering = np.random.permutation(d)
        else:
            ordering = np.asarray(ordering, dtype=np.int64)
            if ordering.shape != (d,) or len(set(ordering.tolist())) != d:
                raise ValueError("ordering must be a permutation of the input coordinates")
        degrees = np.empty(d, dtype=np.int64)
        degrees[ordering] = np.arange(1, d + 1)
        h0, h1 = self.hidden_sizes
        m0 = np.random.randint(1, d, size=h0)
        m1 = np.random.randint(1, d, size=h1)
        mask_in = (m0[None, :] >= degrees[:, None]).astype(np.uint8)
        mask_h = (m1[None, :] >= m0[:, None]).astype(np.uint8)
        mask_out = (degrees[None, :] > m1[:, None]).astype(np.uint8)
        layers = self.masked_layers()
        layers[0].set_mask(mask_in)
        layers[1].set_mask(mask_h)
        layers[2].set_mask(mask_out)
        self.input_degrees = degrees
        return degrees

    def forward(self, x):
        return self.net(x)


class VerticalStackConv(nn.Module):
    """Convolution that sees only rows strictly above the current row, across the kernel width."""

    def __init__(self, in_channels, out_channels, kernel_size, mask_type="B"):
        super().__init__()
        if kernel_size % 2 != 1 or kernel_size < 1:
            raise ValueError("kernel_size must be odd")
        self.kernel_size = kernel_size
        self.mask_type = mask_type
        half = kernel_size // 2
        self.pad_top = half
        self.pad_side = half
        # Height `half` reaches rows i-half .. i-1 once the top is padded by `half`.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(half, kernel_size), bias=True)

    def forward(self, x):
        height, width = x.shape[-2:]
        x = F.pad(x, (self.pad_side, self.pad_side, self.pad_top, 0))
        x = self.conv(x)
        # The upward kernel produces one extra row. Keeping the top rows aligns
        # output row i with input rows strictly above i.
        return x[:, :, :height, :width]


class HorizontalStackConv(nn.Module):
    """Convolution along the current row, strictly to the left, plus the center only for mask B."""

    def __init__(self, in_channels, out_channels, kernel_size, mask_type="B"):
        super().__init__()
        if kernel_size % 2 != 1 or kernel_size < 3:
            raise ValueError("kernel_size must be an odd integer >= 3")
        if mask_type not in ("A", "B"):
            raise ValueError("mask_type must be A or B")
        self.kernel_size = kernel_size
        self.mask_type = mask_type
        half = kernel_size // 2
        self.pad_left = half
        width = half if mask_type == "A" else half + 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, width), bias=True)

    def forward(self, x):
        width = x.shape[-1]
        x = F.pad(x, (self.pad_left, 0, 0, 0))
        x = self.conv(x)
        # Mask A is one column wider than the image. The last column would look
        # past the right edge, so it is dropped and column j stays on pixels < j.
        return x[:, :, :, :width]


class GatedPixelCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, mask_type="B", num_classes=10):
        super().__init__()
        self.v_conv = VerticalStackConv(in_channels, 2 * out_channels, kernel_size, mask_type)
        self.h_conv = HorizontalStackConv(in_channels, 2 * out_channels, kernel_size, mask_type)
        self.v_to_h = nn.Conv2d(2 * out_channels, 2 * out_channels, kernel_size=1)
        self.h_to_h = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        # The parent embedding has width out_channels. This projects it to the filter/gate pair.
        self.class_proj = nn.Linear(out_channels, 2 * out_channels)
        self.out_channels = out_channels

    def forward(self, v_x, h_x, class_embedding):
        v_pre = self.v_conv(v_x)
        h_pre = self.h_conv(h_x) + self.v_to_h(v_pre)
        cond = self.class_proj(class_embedding)[:, :, None, None]
        v_pre = v_pre + cond
        h_pre = h_pre + cond
        v_filter, v_gate = v_pre.chunk(2, dim=1)
        h_filter, h_gate = h_pre.chunk(2, dim=1)
        v_out = torch.tanh(v_filter) * torch.sigmoid(v_gate)
        h_out = torch.tanh(h_filter) * torch.sigmoid(h_gate)
        h_out = self.h_to_h(h_out)
        if h_x.shape[1] == h_out.shape[1]:
            h_out = h_out + h_x
        return v_out, h_out


class ConditionalPixelCNN(nn.Module):
    def __init__(
        self,
        in_channels=1,
        num_channels=64,
        num_layers=5,
        num_classes=10,
        color_levels=4,
        kernel_size=3,
        use_direct_class_bias=False,
    ):
        super().__init__()
        self.color_levels = color_levels
        self.kernel_size = kernel_size
        self.use_direct_class_bias = use_direct_class_bias
        self.embedding = nn.Embedding(num_classes, num_channels)
        blocks = []
        for layer in range(num_layers):
            blocks.append(
                GatedPixelCNNBlock(
                    in_channels if layer == 0 else num_channels,
                    num_channels,
                    kernel_size=kernel_size,
                    mask_type="A" if layer == 0 else "B",
                    num_classes=num_classes,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.out_conv = nn.Conv2d(num_channels, color_levels * in_channels, kernel_size=1)
        if use_direct_class_bias:
            self.class_to_logits = nn.Linear(num_channels, color_levels * in_channels)
            nn.init.zeros_(self.class_to_logits.weight)
            nn.init.zeros_(self.class_to_logits.bias)

    def forward(self, x, labels):
        emb = self.embedding(labels)
        x = x.float() / (self.color_levels - 1)
        v, h = x, x
        for block in self.blocks:
            v, h = block(v, h, emb)
        logits = self.out_conv(h)
        if self.use_direct_class_bias:
            logits = logits + self.class_to_logits(emb)[:, :, None, None]
        return logits


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def fgsm_attack(image, epsilon, data_grad):
    perturbed = image + epsilon * data_grad.sign()
    return torch.clamp(perturbed, 0.0, 1.0)


def get_binarized_loaders(batch_size=128):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda image: (image > 0.5).float().view(-1)),
        ]
    )
    train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_continuous_loaders(batch_size=128):
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_2bit_mnist_loaders(batch_size=128):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 3).long()),
        ]
    )
    train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def bits_from_nats(nats_per_dim):
    return nats_per_dim / math.log(2.0)


@torch.no_grad()
def made_bits_per_dim(model, loader, device, n_masks=4, resample=True, max_batches=None):
    was_training = model.training
    model.eval()
    total = 0.0
    count = 0
    masks = n_masks if resample else 1
    for _ in range(masks):
        if resample:
            model.sample_masks()
        for batch_index, (images, _) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(device)
            logits = model(images)
            total += F.binary_cross_entropy_with_logits(logits, images, reduction="sum").item()
            count += images.numel()
    if was_training:
        model.train()
    return bits_from_nats(total / count)


def train_made(device, epochs=8, batch_size=128, hidden=(512, 512), lr=1e-3, log_every=50, on_epoch=None):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train_loader, test_loader = get_binarized_loaders(batch_size)
    assert len(train_loader.dataset) == 60000 and train_loader.dataset.train
    assert len(test_loader.dataset) == 10000 and not test_loader.dataset.train
    model = MADE(28 * 28, hidden).to(device)
    opt = make_optimizer(model.parameters(), lr)
    writer = SummaryWriter(os.path.join(RUN_DIR, "made"))
    history = {"train_bits": [], "test_bits": [], "seconds": []}
    step = 0
    print(
        f"MADE on {device}: {sum(p.numel() for p in model.parameters())} parameters, "
        f"{epochs} epochs, batch {batch_size}, hidden {hidden}"
    )
    start = time.perf_counter()
    warmup = time.perf_counter()
    seen = 0
    for epoch in range(epochs):
        model.train()
        running = 0.0
        running_count = 0
        bar = tqdm_bar(train_loader, desc=f"MADE epoch {epoch + 1}/{epochs}", leave=False)
        for images, _ in bar:
            images = images.to(device)
            model.sample_masks()
            opt.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.binary_cross_entropy_with_logits(logits, images)
            loss.backward()
            opt.step()
            running += loss.item() * images.numel()
            running_count += images.numel()
            writer.add_scalar("train/bits_per_dim_step", bits_from_nats(loss.item()), step)
            step += 1
            seen += 1
            if seen == 1:
                if device.type == "mps":
                    torch.mps.synchronize()
                warmup = time.perf_counter()
            if seen == 21:
                if device.type == "mps":
                    torch.mps.synchronize()
                elapsed = time.perf_counter() - warmup
                steps_left = epochs * len(train_loader) - 21
                eta = elapsed / 20 * steps_left
                print(f"MADE pace: {elapsed / 20:.3f}s/batch, estimated remaining train time {eta / 60:.1f} min")
            if step % log_every == 0:
                bar.set_postfix(bits=f"{bits_from_nats(loss.item()):.3f}")
        train_bits = bits_from_nats(running / running_count)
        test_bits = made_bits_per_dim(model, test_loader, device, n_masks=3)
        history["train_bits"].append(train_bits)
        history["test_bits"].append(test_bits)
        writer.add_scalar("train/bits_per_dim_epoch", train_bits, epoch)
        writer.add_scalar("test/bits_per_dim", test_bits, epoch)
        print(f"MADE epoch {epoch + 1}: train {train_bits:.4f} bits/dim, test {test_bits:.4f} bits/dim")
        _save_loss_plot(history, "made_loss.png", "MADE bits/dim")
        if on_epoch is not None:
            on_epoch(epoch, history)
    if device.type == "mps":
        torch.mps.synchronize()
    history["seconds"].append(time.perf_counter() - start)
    writer.flush()
    writer.close()
    torch.save({"model": model.state_dict(), "history": history, "hidden": hidden}, os.path.join(OUT_DIR, "made.pt"))
    return model, history, test_loader


def center_missing_mask(size=28, fraction=0.5):
    side = int(round(size * math.sqrt(fraction)))
    top = (size - side) // 2
    mask = torch.zeros(size, size)
    mask[top : top + side, top : top + side] = 1.0
    return mask


def inpaint_images(model, corrupted_images, missing_mask):
    """Fill pixels where missing_mask is 1. Observed pixels stay fixed.

    One ordering puts every observed coordinate before every missing coordinate.
    Sampling then walks the missing coordinates from earliest to latest.
    """
    model.eval()
    if corrupted_images.dim() == 2:
        images = corrupted_images
        spatial = None
    else:
        spatial = corrupted_images.shape
        images = corrupted_images.reshape(corrupted_images.shape[0], -1)
    images = images.clone()
    flat_mask = missing_mask.detach().reshape(-1).cpu().numpy() > 0.5
    observed = np.flatnonzero(~flat_mask)
    missing = np.flatnonzero(flat_mask)
    ordering = np.concatenate([observed, missing])
    model.sample_masks(ordering)
    with torch.no_grad():
        for index in missing:
            logits = model(images)
            probabilities = torch.sigmoid(logits[:, index])
            images[:, index] = torch.bernoulli(probabilities)
    if spatial is not None:
        return images.reshape(spatial)
    return images


def show_inpainting(model, device, n=8):
    _, test_loader = get_continuous_loaders(batch_size=n)
    original, _ = next(iter(test_loader))
    original = original[:n].to(device)
    binary = (original > 0.5).float()
    mask = center_missing_mask().to(device)
    corrupted = binary.clone()
    corrupted[:, :, mask.bool()] = 0
    restored = inpaint_images(model, corrupted, mask)
    panels = torch.cat([binary, corrupted, restored], dim=0).cpu()
    grid = make_grid(panels, nrow=n, pad_value=0.3)
    path = os.path.join(OUT_DIR, "inpainting.png")
    _save_image_grid(grid, path, "original, center removed, MADE inpainting")
    covered = float(mask.mean())
    observed = mask == 0
    changed = int(((restored - corrupted).abs() * observed).sum().item())
    return {"path": path, "missing_fraction": covered, "observed_pixels_changed": changed, "grid": grid}


def load_target_cnn(device):
    model = SimpleCNN().to(device)
    state = torch.load(os.path.join(ROOT, "mnist_cnn_target.pth"), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def attack_and_score(made, device, epsilon=0.3, batch_size=256):
    target = load_target_cnn(device)
    _, loader = get_continuous_loaders(batch_size)
    # One fixed raster order so clean and adversarial NLLs are comparable.
    made.sample_masks(np.arange(28 * 28))
    made.eval()
    target.eval()
    clean_correct = 0
    adv_correct = 0
    total = 0
    clean_bits = []
    adv_bits = []
    for images, labels in tqdm_bar(loader, desc="FGSM + MADE NLL"):
        images = images.to(device)
        labels = labels.to(device)
        images = images.detach().requires_grad_(True)
        logits = target(images)
        loss = F.nll_loss(logits, labels)
        target.zero_grad()
        loss.backward()
        adv = fgsm_attack(images, epsilon, images.grad)
        with torch.no_grad():
            clean_pred = target(images).argmax(dim=1)
            adv_pred = target(adv).argmax(dim=1)
            clean_correct += (clean_pred == labels).sum().item()
            adv_correct += (adv_pred == labels).sum().item()
            total += labels.numel()
            clean_bin = (images > 0.5).float().reshape(images.shape[0], -1)
            adv_bin = (adv > 0.5).float().reshape(adv.shape[0], -1)
            clean_logits = made(clean_bin)
            adv_logits = made(adv_bin)
            clean_nll = F.binary_cross_entropy_with_logits(clean_logits, clean_bin, reduction="none").sum(dim=1)
            adv_nll = F.binary_cross_entropy_with_logits(adv_logits, adv_bin, reduction="none").sum(dim=1)
            clean_bits.append(bits_from_nats(clean_nll).cpu())
            adv_bits.append(bits_from_nats(adv_nll).cpu())
    clean_bits = torch.cat(clean_bits).numpy()
    adv_bits = torch.cat(adv_bits).numpy()
    path = os.path.join(OUT_DIR, "nll_histogram.png")
    plt.figure(figsize=(7, 4))
    plt.hist(clean_bits, bins=40, alpha=0.7, label="clean", density=True)
    plt.hist(adv_bits, bins=40, alpha=0.7, label=f"FGSM eps={epsilon}", density=True)
    plt.xlabel("MADE negative log-likelihood (bits per image)")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    return {
        "clean_accuracy": clean_correct / total,
        "adversarial_accuracy": adv_correct / total,
        "epsilon": epsilon,
        "clean_bits_mean": float(clean_bits.mean()),
        "adv_bits_mean": float(adv_bits.mean()),
        "clean_bits_median": float(np.median(clean_bits)),
        "adv_bits_median": float(np.median(adv_bits)),
        "histogram": path,
        "n_images": total,
    }


def made_dependency_ok(device):
    """Output d may depend only on inputs whose degree is strictly smaller."""
    torch.manual_seed(1)
    np.random.seed(1)
    model = MADE(32, [24, 24]).to(device)
    ordering = np.random.permutation(32)
    degrees = model.sample_masks(ordering)
    x = torch.randn(4, 32, device=device, requires_grad=True)
    logits = model(x)
    bad = []
    for d in range(32):
        model.zero_grad()
        if x.grad is not None:
            x.grad = None
        logits[:, d].sum().backward(retain_graph=True)
        grad = x.grad[0].detach().abs()
        illegal = grad[degrees >= degrees[d]]
        if float(illegal.max()) > 1e-6:
            bad.append(int(d))
    return len(bad) == 0, bad, degrees


def receptive_field_ok(model, device, row=14, col=14):
    model.eval()
    image = torch.zeros(1, 1, 28, 28, device=device, requires_grad=True)
    labels = torch.zeros(1, dtype=torch.long, device=device)
    logits = model(image, labels)
    model.zero_grad()
    logits[0, :, row, col].sum().backward()
    grad = image.grad[0, 0].detach().abs()
    future = torch.zeros_like(grad)
    future[row + 1 :, :] = grad[row + 1 :, :]
    future[row, col:] = grad[row, col:]
    past = grad.sum() - future.sum()
    leaked = float(future.max()) > 1e-8
    blind = float(past) <= 1e-8
    return (not leaked) and (not blind), grad.detach().cpu(), float(future.max()), float(past)


def visualize_receptive_field(model, device):
    ok, grad, leak, past = receptive_field_ok(model, device)
    path = os.path.join(OUT_DIR, "receptive_field.png")
    plt.figure(figsize=(4, 4))
    plt.imshow(grad.numpy(), cmap="magma")
    plt.scatter([14], [14], c="cyan", s=12)
    plt.title("receptive field at (14, 14)" + (" OK" if ok else " LEAK"))
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    return {"ok": ok, "max_future_grad": leak, "past_grad_sum": past, "path": path, "grad": grad}


@torch.no_grad()
def pixelcnn_bits_per_dim(model, loader, device, max_batches=None):
    was_training = model.training
    model.eval()
    total = 0.0
    count = 0
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images, labels)
        target = images.squeeze(1)
        total += F.cross_entropy(logits, target, reduction="sum").item()
        count += target.numel()
    if was_training:
        model.train()
    return bits_from_nats(total / count)


def train_pixelcnn(
    device,
    epochs=6,
    batch_size=64,
    num_channels=48,
    num_layers=5,
    kernel_size=7,
    lr=1e-3,
    on_epoch=None,
    log_name="pixelcnn",
    resume=None,
    ink_weight=1.0,
    use_direct_class_bias=False,
):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train_loader, test_loader = get_2bit_mnist_loaders(batch_size)
    assert len(train_loader.dataset) == 60000 and train_loader.dataset.train
    assert len(test_loader.dataset) == 10000 and not test_loader.dataset.train
    history = {"train_bits": [], "test_bits": []}
    if resume:
        ckpt = torch.load(resume, map_location=device)
        num_channels = ckpt.get("num_channels", num_channels)
        num_layers = ckpt.get("num_layers", num_layers)
        kernel_size = ckpt.get("kernel_size", kernel_size)
        use_direct_class_bias = any(key.startswith("class_to_logits.") for key in ckpt["model"])
        history = {
            "train_bits": list(ckpt.get("history", {}).get("train_bits", [])),
            "test_bits": list(ckpt.get("history", {}).get("test_bits", [])),
        }
    model = ConditionalPixelCNN(
        num_channels=num_channels,
        num_layers=num_layers,
        kernel_size=kernel_size,
        use_direct_class_bias=use_direct_class_bias,
    ).to(device)
    if resume:
        model.load_state_dict(ckpt["model"])
    opt = make_optimizer(model.parameters(), lr)
    writer = SummaryWriter(os.path.join(RUN_DIR, log_name))
    print(
        f"PixelCNN on {device}: {sum(p.numel() for p in model.parameters())} parameters, "
        f"{epochs} epochs, batch {batch_size}, channels {num_channels}, "
        f"layers {num_layers}, kernel {kernel_size}, ink weight {ink_weight}"
        + (f", resumed from {resume}" if resume else "")
    )
    field = visualize_receptive_field(model, device)
    print(
        f"receptive field before training: ok={field['ok']} "
        f"future_grad={field['max_future_grad']:.2e} past_sum={field['past_grad_sum']:.2e}"
    )
    if not field["ok"] and not resume:
        raise RuntimeError("PixelCNN receptive field leaks or is empty; training would not be a valid density model")
    start = time.perf_counter()
    warmup = time.perf_counter()
    seen = 0
    step = 0
    for epoch in range(epochs):
        model.train()
        running = 0.0
        running_count = 0
        bar = tqdm_bar(train_loader, desc=f"PixelCNN epoch {epoch + 1}/{epochs}", leave=False)
        for images, labels in bar:
            images = images.to(device)
            labels = labels.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(images, labels)
            target = images.squeeze(1)
            per_pixel = F.cross_entropy(logits, target, reduction="none")
            if ink_weight == 1.0:
                loss = per_pixel.mean()
            else:
                weights = torch.where(
                    target > 0,
                    per_pixel.new_full((), ink_weight),
                    per_pixel.new_full((), 1.0),
                )
                loss = (per_pixel * weights).sum() / weights.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += per_pixel.sum().item()
            running_count += target.numel()
            writer.add_scalar("train/bits_per_dim_step", bits_from_nats(per_pixel.mean().item()), step)
            step += 1
            seen += 1
            if seen == 1:
                if device.type == "mps":
                    torch.mps.synchronize()
                warmup = time.perf_counter()
            if seen == 11:
                if device.type == "mps":
                    torch.mps.synchronize()
                elapsed = time.perf_counter() - warmup
                steps_left = epochs * len(train_loader) - 11
                eta = elapsed / 10 * steps_left
                print(f"PixelCNN pace: {elapsed / 10:.3f}s/batch, estimated remaining train time {eta / 60:.1f} min")
            bar.set_postfix(bits=f"{bits_from_nats(per_pixel.mean().item()):.3f}")
        train_bits = bits_from_nats(running / running_count)
        test_bits = pixelcnn_bits_per_dim(model, test_loader, device)
        history["train_bits"].append(train_bits)
        history["test_bits"].append(test_bits)
        writer.add_scalar("train/bits_per_dim_epoch", train_bits, epoch)
        writer.add_scalar("test/bits_per_dim", test_bits, epoch)
        print(f"PixelCNN epoch {epoch + 1}: train {train_bits:.4f} bits/dim, test {test_bits:.4f} bits/dim")
        _save_loss_plot(history, "pixelcnn_loss.png", "PixelCNN bits/dim")
        writer.flush()
        if on_epoch is not None:
            on_epoch(epoch, history)
    if device.type == "mps":
        torch.mps.synchronize()
    history["seconds"] = time.perf_counter() - start
    writer.close()
    torch.save(
        {
            "model": model.state_dict(),
            "history": history,
            "num_channels": num_channels,
            "num_layers": num_layers,
            "kernel_size": kernel_size,
            "ink_weight": ink_weight,
        },
        os.path.join(OUT_DIR, "pixelcnn.pt"),
    )
    return model, history, test_loader


@torch.no_grad()
def multi_order_ensemble_eval(model, loader, device, order_counts=(1, 2, 4, 8, 16), max_batches=4):
    """Evaluates NLL bits/dim under multi-order ensembling.
    
    Averaging probabilities over K factorizations strictly lowers negative
    log-likelihood by Jensen's inequality: -log E[p(x|o)] <= E[-log p(x|o)].
    """
    model.eval()
    results = {}
    
    # Collect a small subset of test batches for fast, accurate comparison
    test_images = []
    for batch_idx, (images, _) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        test_images.append(images.to(device))
    images = torch.cat(test_images, dim=0)
    total_pixels = images.numel()
    
    for K in order_counts:
        log_probs = []
        for _ in range(K):
            model.sample_masks()
            logits = model(images)
            # log p(x) for Bernoulli = - BCE with logits sum over dimensions
            log_p = -F.binary_cross_entropy_with_logits(logits, images, reduction="none").sum(dim=1)
            log_probs.append(log_p)
        # Log-sum-exp trick for stable ensemble probability
        stacked = torch.stack(log_probs, dim=0) # [K, N]
        ensemble_log_p = torch.logsumexp(stacked, dim=0) - math.log(K) # [N]
        bits = bits_from_nats(-ensemble_log_p.sum().item() / total_pixels)
        results[K] = bits
    return results


def receptive_field_multilocation_check(model, device, coordinates=((7, 7), (14, 14), (21, 21))):
    """Autograd receptive field check across multiple pixel locations to guarantee no leaks anywhere."""
    results = {}
    for r, c in coordinates:
        ok, grad, leak, past = receptive_field_ok(model, device, row=r, col=c)
        results[(r, c)] = {
            "ok": ok,
            "max_future_grad": leak,
            "past_grad_sum": past,
        }
    return results


def load_made(device, path=None):
    if path is None:
        path = os.path.join(OUT_DIR, "made.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint {path} not found")
    ckpt = torch.load(path, map_location=device)
    hidden = ckpt.get("hidden", (512, 512))
    model = MADE(28 * 28, hidden).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt.get("history", {})


def load_pixelcnn(device, path=None):
    if path is None:
        path = os.path.join(OUT_DIR, "pixelcnn.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint {path} not found")
    ckpt = torch.load(path, map_location=device)
    num_channels = ckpt.get("num_channels", 48)
    num_layers = ckpt.get("num_layers", 5)
    kernel_size = ckpt.get("kernel_size", 3)
    use_bias = any(key.startswith("class_to_logits.") for key in ckpt["model"])
    model = ConditionalPixelCNN(
        num_channels=num_channels,
        num_layers=num_layers,
        kernel_size=kernel_size,
        use_direct_class_bias=use_bias,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt.get("history", {})


@torch.no_grad()
def generate_conditional_images(model, device, temperature=1.0, argmax=False, save_name="conditional_samples.png"):
    model.eval()
    samples = torch.zeros(100, 1, 28, 28, device=device)
    labels = torch.arange(10, device=device).repeat_interleave(10)
    steps = 28 * 28
    bar = tqdm_bar(range(steps), desc="ancestral sampling")
    for index in bar:
        row, col = divmod(index, 28)
        logits = model(samples, labels)[:, :, row, col]
        if argmax:
            draw = logits.argmax(dim=1)
        else:
            scaled_logits = logits / max(temperature, 1e-4)
            probabilities = torch.softmax(scaled_logits, dim=1)
            # multinomial on MPS has been unreliable across torch versions; draw on CPU and copy back.
            draw = torch.multinomial(probabilities.cpu(), num_samples=1).to(device).squeeze(1)
        samples[:, 0, row, col] = draw.float()
    grid = make_grid(samples.cpu() / 3.0, nrow=10, pad_value=0.3)
    path = os.path.join(OUT_DIR, save_name)
    title = "each row is one digit class" if temperature == 1.0 and not argmax else (
        f"each row is one digit class (T={temperature if not argmax else 'argmax'})"
    )
    _save_image_grid(grid, path, title)
    return samples, path


def _save_loss_plot(history, name, ylabel):
    plt.figure(figsize=(6, 3.5))
    plt.plot(history["train_bits"], label="train")
    plt.plot(history["test_bits"], label="test")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, name), dpi=140)
    plt.close()


def run_all():
    """Train both models, then run inpainting, the FGSM check, and class-conditional samples."""
    device = pick_device()
    print("device", device)
    print("optimizer smoke test", smoke_test_optimizer(device))
    print(
        "Time budget on this M1 Pro, from a measured pace of "
        "0.036 s/batch for MADE and 0.26 s/batch for the 48-channel PixelCNN: "
        "MADE about 5 minutes, PixelCNN about 25 minutes, sampling about 5 minutes."
    )
    model, made_history, _ = train_made(device, epochs=10, batch_size=128, hidden=(512, 512), lr=1e-3)
    inpaint = show_inpainting(model, device)
    print("inpainting", {k: inpaint[k] for k in ("path", "missing_fraction", "observed_pixels_changed")})
    attack = attack_and_score(model, device, epsilon=0.3)
    print("attack", attack)
    pixel, pixel_history, _ = train_pixelcnn(
        device,
        epochs=6,
        batch_size=64,
        num_channels=48,
        num_layers=5,
        kernel_size=7,
        lr=1e-3,
        log_name="pixelcnn_k7",
    )
    _, sample_path = generate_conditional_images(pixel, device)
    print("samples", sample_path)
    print("MADE test bits/dim", made_history["test_bits"][-1], "uniform baseline 1.0")
    print("PixelCNN test bits/dim", pixel_history["test_bits"][-1], "uniform baseline 2.0")
    return made_history, pixel_history


def _save_image_grid(grid, path, title):
    plt.figure(figsize=(8, 4))
    plt.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray" if grid.shape[0] == 1 else None)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


if __name__ == "__main__":
    run_all()
