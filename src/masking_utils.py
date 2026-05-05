import inflect
import cv2
import numpy as np
from typing import List
from PIL import Image

def calculate_token_coverage(mask, grid_h, grid_w):
    """Calculate coverage ratio for every token cell across the full grid."""
    H, W = mask.shape
    patch_h = H // grid_h
    patch_w = W // grid_w
    coverage_map = np.zeros((grid_h, grid_w))
    for r in range(grid_h):
        for c in range(grid_w):
            patch = mask[r*patch_h:(r+1)*patch_h, c*patch_w:(c+1)*patch_w]
            coverage_map[r, c] = np.sum(patch > 0) / (patch_h * patch_w)
    return coverage_map


def get_mask_and_bb_positions(mask_path, json_data, image_positions, grid_size=15, threshold=0.1):
    """
    Args:
        mask_path: path to the binary mask image
        json_data: the dict for one mask entry (e.g. json["img0_mask_0"])
        image_positions: list of absolute positions of image tokens in input_ids
        grid_size: grid size used for tokenization (15)
        threshold: coverage threshold (0.1)
    Returns:
        mask_positions:    positions for tokens covering the mask (coverage > threshold)
        bb_positions:      positions for ALL tokens inside ALL bounding box rectangles
        bb_mask_positions: positions for mask tokens within ALL BBs (from token_indices)
    """
    grid_key = f"grid_{grid_size}x{grid_size}"
    grid_boxes = json_data[grid_key]            # list of boxes
    grid_h, grid_w = grid_boxes[0]["token_grid"]
    token_h = 480 // grid_h
    token_w = 480 // grid_w
    image_start = image_positions[0]

    # --- Mask positions: load mask, compute coverage, threshold ---
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    expected_h = grid_h * (mask.shape[0] // grid_h)
    expected_w = grid_w * (mask.shape[1] // grid_w)
    if mask.shape[0] != expected_h or mask.shape[1] != expected_w:
        mask = cv2.resize(mask, (expected_w, expected_h))
    coverage_map = calculate_token_coverage(mask, grid_h, grid_w)
    mask_token_coords = np.argwhere(coverage_map > threshold)   # [row, col] pairs
    mask_positions = [
        int(image_start + (row * grid_w + col))
        for row, col in mask_token_coords
    ]

    # --- BB positions: ALL tokens inside ALL bounding box rectangles ---
    bb_positions = set()
    for box_data in grid_boxes:
        x1, y1, x2, y2 = box_data["scaled_selection"]
        row_start = y1 // token_h
        row_end   = (y2 - 1) // token_h
        col_start = x1 // token_w
        col_end   = (x2 - 1) // token_w
        for row in range(row_start, row_end + 1):
            for col in range(col_start, col_end + 1):
                bb_positions.add(image_start + (row * grid_w + col))
    bb_positions = list(bb_positions)

    # --- BB mask positions: token_indices across ALL boxes ---
    bb_mask_positions = set()
    for box_data in grid_boxes:
        for row, col in box_data["token_indices"]:
            bb_mask_positions.add(image_start + (row * grid_w + col))
    bb_mask_positions = list(bb_mask_positions)

    return mask_positions, bb_positions, bb_mask_positions

def mask_image_pixels(image_path: str, mask_path: str, fill_value: int = 0):
    """
    Mask image pixels using a binary mask (full mask).
    Args:
        image_path: Path to the original image
        mask_path: Path to the binary mask
        fill_value: Value to fill masked pixels with (default: 0 for black)
    Returns:
        PIL Image with masked pixels
    """
    # Load and convert to numpy
    image = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))

    # Apply mask
    image[mask > 0] = fill_value

    return Image.fromarray(image)

def mask_image_pixels_bb(
    image_path: str,
    mask_path: str,
    bbox_coords: List[int],
    fill_value: int = 0
):
    """
    Mask image pixels using only the bounding box region of the binary mask.

    Args:
        image_path: Path to the original image
        mask_path: Path to the binary mask
        bbox_coords: Bounding box coordinates [x1, y1, x2, y2] from scaled_selection
        fill_value: Value to fill masked pixels with (default: 0 for black)

    Returns:
        PIL Image with masked pixels (only in BB region)
    """
    # Load image and mask
    image = Image.open(image_path).convert('RGB') if isinstance(image_path, str) else image_path.convert('RGB')
    mask = Image.open(mask_path).convert('L')

    # Convert to numpy arrays
    image_np = np.array(image)
    mask_np = np.array(mask)

    # Extract bounding box coordinates
    x1, y1, x2, y2 = bbox_coords

    # Crop the binary mask to the BB region
    mask_bb_cropped = mask_np[y1:y2, x1:x2]

    # Create binary mask
    mask_bb_binary = mask_bb_cropped > 0

    # Apply the cropped mask to the corresponding BB region in the image
    masked_image = image_np.copy()
    masked_image[y1:y2, x1:x2][mask_bb_binary] = fill_value

    # Convert back to PIL Image
    return Image.fromarray(masked_image.astype(np.uint8))


def mask_image_pixels_bb_full(
        image_path: str,
        bbox_coords: List[int],
        fill_value: int = 0
):
    """
    Mask ALL pixels inside the bounding box rectangle (ignoring the mask).

    Args:
        image_path: Path to the original image
        bbox_coords: Bounding box coordinates [x1, y1, x2, y2]
        fill_value: Value to fill masked pixels with (default: 0 for black)

    Returns:
        PIL Image with entire BB region blacked out
    """
    image = Image.open(image_path).convert('RGB') if isinstance(image_path, str) else image_path.convert('RGB')
    image_np = np.array(image)

    x1, y1, x2, y2 = bbox_coords

    # Black out the ENTIRE BB rectangle
    masked_image = image_np.copy()
    masked_image[y1:y2, x1:x2] = fill_value

    return Image.fromarray(masked_image.astype(np.uint8))