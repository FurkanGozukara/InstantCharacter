# --- START OF REVISED FILE secourses_app.py ---

import torch
import random
import numpy as np
import os
import time # Added for unique filenames
import platform # Added for opening folder
import subprocess # Added for opening folder
import glob # Added for finding LoRA files
import gc # Added for garbage collection
from PIL import Image
import argparse # For command-line arguments

import gradio as gr
from huggingface_hub import hf_hub_download
from transformers import AutoModelForImageSegmentation
from torchvision import transforms

# Ensure pipeline module is accessible (e.g., in the same directory or Python path)
try:
    from pipeline import InstantCharacterFluxPipeline
except ImportError:
    print("Error: 'pipeline.py' not found. Please ensure it's in the same directory or your Python path.")
    exit()

# Import LoRA baking function from user-provided utils.py
try:
    from models.utils import flux_load_lora as bake_lora_into_pipe
    print("Successfully imported flux_load_lora from models.utils.py for LoRA baking.")
except ImportError:
    print("Error: 'models/utils.py' not found or 'flux_load_lora' not defined in it. LoRA baking will not work.")
    print("Please ensure models/utils.py with the required flux_load_lora function is in the same directory.")
    # Define a dummy function to prevent crashes if utils.py is missing, but warn user.
    def bake_lora_into_pipe(pipe, lora_file_path, lora_weight):
        gr.Error("LoRA baking function (flux_load_lora from models/utils.py) not found or failed! Cannot apply LoRA styles. Please ensure models/utils.py is correct and available. Proceeding without this LoRA.")
        print("LoRA baking function (flux_load_lora from models/utils.py) not found or failed! Cannot apply LoRA styles. Please ensure models/utils.py is correct and available. Proceeding without this LoRA.")
        # Return the pipe unmodified, or handle as appropriate if pipe should not be used.
        # For now, returning the pipe allows the flow to continue with the base model.
        return pipe


TORCH_CACHE_DIR = os.path.join(os.getcwd(), "torch_cache") # Define cache directory for torch.compile
import torch._inductor.config as inductor_config

# Add this before the torch.compile code
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
inductor_config.triton.cudagraphs = False
# inductor_config.triton.max_block = 1024  # Reduce block size to avoid shared memory issues

# --- Global Variables and Setup ---
MAX_SEED = np.iinfo(np.int32).max
OUTPUT_DIR = "outputs"
LORAS_DIR = "loras"


# Add after line 66 (after inductor_config settings)
inductor_config.max_autotune = False  # Use a simpler autotuning approach

# In the initialize_pipeline_and_models function
# Modify the compile_kwargs around line 241 to:
compile_kwargs = {
    "fullgraph": False, 
    "dynamic": True, 
    "mode": "reduce-overhead",  # Change from max-autotune to reduce-overhead
    "backend": "inductor"
}

args = None # Will be populated by argparse
pipe = None # Global pipeline instance
current_baked_lora_path = None # Tracks the currently baked LoRA
birefnet = None # Matting model
birefnet_transform_image = None # Matting model transforms

# Model paths (will be used in initialize_pipeline_and_models)
ip_adapter_path = None
base_model = 'MonsterMMORPG/flux_dev_backup'
image_encoder_path_g = 'google/siglip-so400m-patch14-384' # Renamed to avoid conflict
image_encoder_2_path_g = 'facebook/dinov2-giant' # Renamed
birefnet_path_g = 'ZhengPeng7/BiRefNet'

# --- Ensure essential directories exist early ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LORAS_DIR, exist_ok=True)
os.makedirs("assets", exist_ok=True) # Ensure assets dir for examples

# --- Download default LoRAs (moved from __main__) ---
# This ensures LoRAs are available before get_available_loras() is called for UI setup
lora_files_to_download_on_startup = {
    "Makoto_Shinkai_style.safetensors": "InstantX/FLUX.1-dev-LoRA-Makoto-Shinkai",
    "ghibli_style.safetensors": "InstantX/FLUX.1-dev-LoRA-Ghibli",
    "Ghibli_Anime_Art_Style.safetensors": "BestModelsv2/flux_loras"
}
import shutil # Ensure shutil is imported if not already at the top
for filename, repo_id in lora_files_to_download_on_startup.items():
    local_path = os.path.join(LORAS_DIR, filename)
    if not os.path.exists(local_path):
        print(f"Downloading {filename} from {repo_id} to {LORAS_DIR} for initial setup...")
        try:
            downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename)
            shutil.copy(downloaded_path, local_path)
        except Exception as e:
            print(f"Failed to download initial LoRA {filename}: {e}")


# Determine device and dtype (can be overridden by args)
device_str = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    dtype_torch = torch.bfloat16
elif torch.cuda.is_available():
    dtype_torch = torch.float16
else:
    dtype_torch = torch.float32


def setup_global_paths():
    global ip_adapter_path, base_model, image_encoder_path_g, image_encoder_2_path_g, birefnet_path_g
    print("Downloading/Loading model component paths...")
    try:
        ip_adapter_path = hf_hub_download(repo_id="Tencent/InstantCharacter", filename="instantcharacter_ip-adapter.bin")
        # base_model, image_encoder_path_g, etc. are already defined
    except Exception as e:
        print(f"Error downloading initial model component (IP Adapter): {e}")
        gr.Error(f"Failed to download IP Adapter: {e}. Check connection and Hugging Face Hub access.")
        exit()
    print("Finished downloading/loading model component paths.")


def initialize_pipeline_and_models(lora_file_path_to_bake=None):
    global pipe, current_baked_lora_path, args, device_str, dtype_torch
    global ip_adapter_path, base_model, image_encoder_path_g, image_encoder_2_path_g
    # Matting model is loaded separately and once, see main block

    # --- Delete existing pipeline if it exists ---
    if pipe is not None:
        print("Deleting existing pipeline instance...")
        # To assist garbage collection, explicitly delete components if they are large
        # or if `torch.compile` might hold references.
        # This is a best-effort, `del pipe` should be the main mechanism.
        # components_to_delete = ['transformer', 'vae', 'text_encoder', 'text_encoder_2',
        #                         'siglip_image_encoder', 'dino_image_encoder_2', 'subject_image_proj_model']
        # for comp_name in components_to_delete:
        #     if hasattr(pipe, comp_name):
        #         delattr(pipe, comp_name)
        del pipe
        pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Existing pipeline deleted and CUDA cache cleared.")

    print(f"Loading base model: {base_model}")
    pipe = InstantCharacterFluxPipeline.from_pretrained(base_model, torch_dtype=dtype_torch)
    print(f"Base model {base_model} loaded.")

    # --- Bake LoRA if specified ---
    if lora_file_path_to_bake and os.path.exists(lora_file_path_to_bake):
        print(f"Attempting to bake LoRA weights from: {lora_file_path_to_bake}")
        try:
            bake_lora_into_pipe(pipe, lora_file_path_to_bake, lora_weight=1.0) # Standard weight
            print(f"LoRA weights from {lora_file_path_to_bake} baked successfully.")
        except Exception as e:
            print(f"Error baking LoRA '{lora_file_path_to_bake}': {e}")
            gr.Warning(f"Failed to bake LoRA: {lora_file_path_to_bake}. Proceeding with base/previous model state.")
            # If baking fails, current_baked_lora_path should not be updated to this failed LoRA
            # The pipeline `pipe` will have the base model weights.
            lora_file_path_to_bake = None # Mark as not baked for current_baked_lora_path update
    elif lora_file_path_to_bake:
        print(f"Warning: LoRA file not found: {lora_file_path_to_bake}. Proceeding without baking this LoRA.")
        gr.Warning(f"LoRA file not found: {lora_file_path_to_bake}. Using base/previous model state.")
        lora_file_path_to_bake = None # Mark as not baked

    # --- CPU Offloading (if not --highvram) ---
    if not args.highvram:
        print("Applying CPU offloading...")
        pipe.to("cpu") # Move all to CPU first

        pipe._exclude_from_cpu_offload.clear()
        pipe._exclude_layer_from_cpu_offload.clear()

        # Configuration from offload_infer_demo.py
        pipe._exclude_from_cpu_offload.extend([
            'text_encoder', # Text encoder 1 (CLIP-L) kept on GPU
            # vae, text_encoder_2 (T5) will be offloaded by default
        ])
        pipe._exclude_layer_from_cpu_offload.extend([
            "transformer.pos_embed",
            "transformer.time_text_embed",
            "transformer.context_embedder",
            "transformer.x_embedder",
            "transformer.transformer_blocks",    # Main transformer blocks kept on GPU
            # "transformer.single_transformer_blocks", # These will be offloaded as per demo
            "transformer.norm_out",
            "transformer.proj_out",
        ])
        pipe.enable_sequential_cpu_offload(device=torch.device(device_str)) # device_str is 'cuda' or 'cpu'
        print("CPU offloading enabled.")
    else:
        print(f"High VRAM mode. Moving pipeline to {device_str}.")
        pipe.to(device_str)

    # --- Initialize Adapters ---
    # Adapters (image encoders, projector) are initialized after offloading/device placement of the main pipe.
    # The `init_adapter` method should handle placing its own new modules correctly.
    # The `device` argument in `subject_ipadapter_cfg` tells `init_ccp_and_attn_processor` where to put new attn_procs.
    print("Initializing adapters...")
    pipe.init_adapter(
        image_encoder_path=image_encoder_path_g,
        image_encoder_2_path=image_encoder_2_path_g,
        subject_ipadapter_cfg=dict(
            subject_ip_adapter_path=ip_adapter_path,
            nb_token=1024
            # REMOVE: device=torch.device(device_str) # <-- Remove this line
        ),
        device=torch.device(device_str) # Overall device context for adapter init
    )
    print("Adapters initialized.")

    # --- Torch Compile (if --compile_model) ---
    if args.compile_model:
        print("Applying torch.compile to model components...")
        # Dynamo configs are set globally if args.compile_model is true (see main block)
        # torch._dynamo.reset() # Already done globally or not needed repeatedly

        compile_kwargs = {"fullgraph": True, "dynamic": True, "mode": "max-autotune", "backend": "inductor"}
        compile_kwargs = {
    "fullgraph": False, 
    "dynamic": True, 
    "mode": "reduce-overhead",  # Change from max-autotune to reduce-overhead
    "backend": "inductor"
}
        
        # Compile Attn Processors (should be on GPU after init_adapter)
        if hasattr(pipe, 'transformer') and hasattr(pipe.transformer, 'attn_processors'):
            print("Compiling attention processors...")
            for name in list(pipe.transformer.attn_processors.keys()):
                processor = pipe.transformer.attn_processors[name]
                processor.to(torch.device(device_str)) # Ensure on target device
                try:
                    pipe.transformer.attn_processors[name] = torch.compile(processor, **compile_kwargs)
                except Exception as e:
                    print(f"Failed to compile attn_processor {name}: {e}")
                    gr.Warning(f"Compilation failed for attn_processor {name}.")
        
        # Compile parts of the transformer
        # single_transformer_blocks are offloaded (if not highvram), transformer_blocks are on GPU.
        # Compiling offloaded modules: torch.compile handles modules that are moved between CPU and GPU by Accelerate's hooks.
        if hasattr(pipe, 'transformer') and hasattr(pipe.transformer, 'single_transformer_blocks'):
            print("Attempting to compile transformer.single_transformer_blocks as a single unit.")
            try:
                # Use compile() method directly on the module instead of torch.compile()
                pipe.transformer.single_transformer_blocks.compile(
                    fullgraph=True,
                    dynamic=True,
                    mode="max-autotune",
                    backend='inductor'
                )
            except Exception as e:
                print(f"Failed to compile single_transformer_blocks as a single unit: {e}")
                gr.Warning(f"Compilation failed for transformer.single_transformer_blocks (single unit).")


        if hasattr(pipe, 'transformer') and hasattr(pipe.transformer, 'transformer_blocks'):
            print("Attempting to compile transformer.transformer_blocks as a single unit.")
            try:
                pipe.transformer.transformer_blocks.to(torch.device(device_str)) # Ensure on GPU before compile
                # Use compile() method directly on the module instead of torch.compile()
                pipe.transformer.transformer_blocks.compile(
                    fullgraph=True,
                    dynamic=True,
                    mode="max-autotune",
                    backend='inductor'
                )
            except Exception as e:
                print(f"Failed to compile transformer_blocks as a single unit: {e}")
                gr.Warning(f"Compilation failed for transformer.transformer_blocks (single unit).")
        
        # VAE (offloaded if not highvram)
        if hasattr(pipe, 'vae'):
            print("Compiling VAE...")
            try:
                pipe.vae = torch.compile(pipe.vae, **compile_kwargs)
            except Exception as e:
                print(f"Failed to compile VAE: {e}")
                gr.Warning("Compilation failed for VAE.")

        # Text Encoder 1 (on GPU if not highvram)
        if hasattr(pipe, 'text_encoder'):
            print("Compiling Text Encoder 1...")
            try:
                pipe.text_encoder.to(torch.device(device_str)) # Ensure on GPU
                pipe.text_encoder = torch.compile(pipe.text_encoder, **compile_kwargs)
            except Exception as e:
                print(f"Failed to compile Text Encoder 1: {e}")
                gr.Warning("Compilation failed for Text Encoder 1.")

        # Text Encoder 2 (T5, offloaded if not highvram)
        if hasattr(pipe, 'text_encoder_2'):
            print("Compiling Text Encoder 2...")
            # Removed torch.compile for text_encoder_2 to avoid conflict with accelerate offloading
            # try:
            #     pipe.text_encoder_2 = torch.compile(pipe.text_encoder_2, **compile_kwargs)
            # except Exception as e:
            #     print(f"Failed to compile Text Encoder 2: {e}")
            #     gr.Warning("Compilation failed for Text Encoder 2.")
            print("Skipping torch.compile for Text Encoder 2; it will be offloaded if not in --highvram mode.")
        print("torch.compile applied where possible.")

    current_baked_lora_path = lora_file_path_to_bake # Update global state AFTER potential baking
    print(f"Pipeline initialization complete. Current baked LoRA: {current_baked_lora_path if current_baked_lora_path else 'None'}")


def load_matting_model():
    global birefnet, birefnet_transform_image, birefnet_path_g, device_str
    if birefnet is None:
        print(f"Loading matting model from {birefnet_path_g}...")
        try:
            birefnet = AutoModelForImageSegmentation.from_pretrained(birefnet_path_g, trust_remote_code=True)
            birefnet.to(device_str) # Revert to device_str for GPU/CPU based on availability
            birefnet.eval()
            birefnet_transform_image = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            print("Matting model loaded successfully.")
        except Exception as e:
            print(f"Error loading matting model: {e}")
            gr.Error(f"Failed to load matting model: {e}")
            exit() # Critical for app function


# --- LoRA Management Functions ---
def get_available_loras():
    lora_files = glob.glob(os.path.join(LORAS_DIR, "*.safetensors"))
    # Choices format: list of (display_name, value) tuples.
    # Here, display_name and value are the same.
    lora_choices = [("None", "None")]
    lora_mapping = {} # Maps value (which is display_name) to path
    for lora_path in lora_files:
        filename = os.path.basename(lora_path)
        name_no_ext = os.path.splitext(filename)[0]
        display_name = name_no_ext.replace('_', ' ').title()
        lora_choices.append((display_name, display_name))
        lora_mapping[display_name] = lora_path
    return lora_choices, lora_mapping

available_loras_g, lora_path_mapping_g = get_available_loras() # Populate globals for UI and runtime

def refresh_loras():
    global available_loras_g, lora_path_mapping_g
    available_loras_g, lora_path_mapping_g = get_available_loras()
    
    # Count actual LoRA files (excluding "None")
    num_actual_loras = 0
    for display_name, value in available_loras_g:
        if value != "None":
            num_actual_loras +=1
    print(f"Refreshed LoRA list: Found {num_actual_loras} LoRA files in {LORAS_DIR}")

    # Determine the default value (string)
    default_lora_value = "None" # Default to "None"
    has_none_choice = any(val == "None" for _, val in available_loras_g)

    if not has_none_choice and available_loras_g: # If "None" is not an option and list is not empty
        default_lora_value = available_loras_g[0][1] # Use the value of the first actual LoRA
    elif not available_loras_g: # If list is empty (e.g. get_available_loras returned empty)
        default_lora_value = None # Or handle as error, but dropdown needs a value

    return gr.update(choices=available_loras_g, value=default_lora_value)

def open_loras_folder():
    print(f"Opening LoRAs folder: {LORAS_DIR}")
    try:
        abs_path = os.path.abspath(LORAS_DIR)
        if platform.system() == "Windows":
            os.startfile(abs_path)
        elif platform.system() == "Darwin":
            subprocess.run(["open", abs_path], check=True)
        else:
            subprocess.run(["xdg-open", abs_path], check=True)
    except Exception as e:
        print(f"Error opening LoRAs folder: {e}")
        gr.Warning(f"Could not open LoRAs folder: {e}")

# --- Helper Functions (remove_bkg, get_example, save_image, open_folder) ---
# These functions are largely unchanged from the original provided secourses_app.py
# but will use global `birefnet` and `birefnet_transform_image`.

def remove_bkg(subject_image: Image.Image) -> Image.Image:
    global birefnet, birefnet_transform_image, device_str

    # Load matting model if it's not already loaded
    if birefnet is None or birefnet_transform_image is None:
        print("Matting model not loaded or transformers missing, attempting to load...")
        load_matting_model() # This will load it onto device_str (GPU if available)
        if birefnet is None or birefnet_transform_image is None: # Check again after load attempt
            gr.Error("Matting model could not be loaded. Cannot remove background.")
            raise RuntimeError("Matting model failed to load for background removal.")

    if subject_image is None:
        raise ValueError("Input image cannot be None for background removal.")
    # The explicit check for birefnet is None or birefnet_transform_image is None is now at the top
    # So no need to repeat it here directly.
    print("Processing image for background removal...")
    img_pil = subject_image.convert("RGB")

    input_images = birefnet_transform_image(img_pil).unsqueeze(0).to(device_str)
    with torch.no_grad():
        output = birefnet(input_images)
        if isinstance(output, (list, tuple)):
            preds = output[-1].sigmoid().cpu() if not hasattr(output, 'logits') else output.logits.sigmoid().cpu()
        elif hasattr(output, 'logits'):
             preds = output.logits.sigmoid().cpu()
        else:
             preds = output.sigmoid().cpu()

    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(img_pil.size)
    mask_np = np.array(mask)[..., None]

    def get_bbox_from_mask(mask_arr, th=128):
        rows, cols = np.where(mask_arr[:, :, 0] >= th)
        if len(rows) == 0 or len(cols) == 0:
            return [0, 0, mask_arr.shape[1] - 1, mask_arr.shape[0] - 1]
        y1, y2 = np.min(rows), np.max(rows)
        x1, x2 = np.min(cols), np.max(cols)
        height, width = mask_arr.shape[:2]
        return [np.clip(x1, 0, width - 1).round().astype(np.int32),
                np.clip(y1, 0, height - 1).round().astype(np.int32),
                np.clip(x2, 0, width - 1).round().astype(np.int32),
                np.clip(y2, 0, height - 1).round().astype(np.int32)]

    x1, y1, x2, y2 = get_bbox_from_mask(mask_np)

    if x1 >= x2 or y1 >= y2:
        subject_image_np_orig = np.array(img_pil)
        # pad_to_square expects HWC, ensure it if needed
        if subject_image_np_orig.ndim == 2: # Grayscale
            subject_image_np_orig = np.stack((subject_image_np_orig,)*3, axis=-1)
        elif subject_image_np_orig.shape[2] == 4: # RGBA
            subject_image_np_orig = subject_image_np_orig[..., :3] # Use RGB
        
        subject_image_np = pad_to_square(subject_image_np_orig)
        return Image.fromarray(subject_image_np.astype(np.uint8))


    subject_image_np = np.array(img_pil)
    alpha_mask = (mask_np > 128).astype(np.uint8) * 255
    rgba_image = np.concatenate((subject_image_np, alpha_mask), axis=2)
    crop_rgba_image = rgba_image[y1:y2, x1:x2, :]
    
    h_crop, w_crop = crop_rgba_image.shape[:2]
    if h_crop == 0 or w_crop == 0: # Check for empty crop
        subject_image_np_orig = np.array(img_pil)
        if subject_image_np_orig.ndim == 2: subject_image_np_orig = np.stack((subject_image_np_orig,)*3, axis=-1)
        elif subject_image_np_orig.shape[2] == 4: subject_image_np_orig = subject_image_np_orig[..., :3]
        subject_image_np = pad_to_square(subject_image_np_orig)
        return Image.fromarray(subject_image_np.astype(np.uint8))

    white_bkg = np.ones((h_crop, w_crop, 3), dtype=np.uint8) * 255
    alpha = crop_rgba_image[:, :, 3:] / 255.0
    rgb = crop_rgba_image[:, :, :3]
    composite_image = (rgb * alpha + white_bkg * (1 - alpha)).astype(np.uint8)

    def pad_to_square(image, pad_value=255):
        if image.ndim != 3 or image.shape[2] not in [1, 3, 4]: # Allow single channel, RGB, RGBA
             print(f"Warning: Unexpected image shape {image.shape} for padding.")
             if image.ndim == 2: image = np.stack([image]*3, axis=-1) # Grayscale to RGB
             elif image.ndim == 3 and image.shape[2] == 1: image = np.concatenate([image]*3, axis=-1) # Single channel to RGB
             elif image.ndim == 3 and image.shape[2] == 4: image = image[...,:3] # RGBA to RGB for padding logic
             else: raise ValueError(f"Cannot pad image with shape {image.shape}")
        
        if image.shape[2] == 4: # If it's still RGBA (e.g. passed directly)
            image = image[...,:3] # Use RGB for padding calculation base

        H, W, C = image.shape
        if H == W: return image
        diff = abs(H - W)
        pad1, pad2 = diff // 2, diff - (diff // 2)
        pad_width = ((pad1, pad2), (0, 0), (0, 0)) if H < W else ((0, 0), (pad1, pad2), (0, 0))
        return np.pad(image, pad_width, 'constant', constant_values=pad_value)

    crop_pad_obj_image = pad_to_square(composite_image, 255)
    print("Background removal and processing complete.")

    # Unload matting model after use
    print("Unloading matting model to free resources...")
    del birefnet
    birefnet = None # Set global to None so it can be reloaded
    # birefnet_transform_image can remain as it's small and doesn't hold GPU memory
    gc.collect()
    if device_str == "cuda":
        torch.cuda.empty_cache()
    print("Matting model unloaded.")

    return Image.fromarray(crop_pad_obj_image.astype(np.uint8))

def get_example():
    examples = []
    base_examples = [
        ["assets/boy2.jpg", "A man is playing a guitar in street, detailed illustration", 0.9, 'Makoto Shinkai Style'],
        ["assets/boy.jpg", "A man is riding a bike in snow, cinematic lighting", 0.9, 'Makoto Shinkai Style'],
        ["assets/boy2.jpg", "A man is reading a book under a large tree, Ghibli style", 1.0, 'Ghibli Style'],
        ["assets/boy.jpg", "A man in autumn landscape with falling leaves, dreamy atmosphere", 1.0, 'Ghibli Anime Art Style'],
        ["assets/boy.jpg", "photo of a man holding a camera", 1.1, 'None'],
    ]
    for ex in base_examples:
        if os.path.exists(ex[0]): examples.append(ex)
        else: print(f"Skipping example, file not found: {ex[0]}")
    return examples

def save_image(image: Image.Image, seed: int, index: int, prompt: str = "output") -> str:
    timestamp = int(time.time())
    safe_prompt = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in prompt[:30]).rstrip()
    filename = f"{OUTPUT_DIR}/img_{seed}_{index}_{timestamp}_{safe_prompt}.png"
    try:
        image.save(filename)
        print(f"Saved image: {filename}")
        return filename
    except Exception as e:
        print(f"Error saving image {filename}: {e}")
        return None

def open_folder_outputs():
    open_folder_general(OUTPUT_DIR)

def open_folder_general(folder_path):
    print(f"Attempting to open folder: {folder_path}")
    if not os.path.isdir(folder_path):
        gr.Warning(f"Folder '{folder_path}' does not exist.")
        return
    abs_path = os.path.abspath(folder_path)
    try:
        if platform.system() == "Windows": os.startfile(abs_path)
        elif platform.system() == "Darwin": subprocess.run(["open", abs_path], check=True)
        else: subprocess.run(["xdg-open", abs_path], check=True)
    except Exception as e:
        gr.Error(f"Failed to open folder: {e}")

# --- Main Generation Logic ---
@torch.inference_mode() # Important for generation
def run_generation_loop(input_image,
                        prompt_text, # Renamed to avoid conflict with prompt module
                        scale,
                        guidance_scale,
                        num_inference_steps,
                        seed,
                        randomize_seed,
                        style_mode,
                        num_generations):
    global pipe, current_baked_lora_path, lora_path_mapping_g, device_str

    if input_image is None:
        gr.Warning("Input image not provided!")
        return [], seed

    if pipe is None: # Should be initialized at startup, but as a fallback
        gr.Info("Pipeline not initialized. Initializing now...")
        initialize_pipeline_and_models() # Initialize with no LoRA
        if pipe is None: # If still None after attempt
            gr.Error("Critical Error: Pipeline failed to initialize!")
            return [], seed


    # --- LoRA Change Detection and Pipeline Re-initialization ---
    selected_lora_display_name = style_mode
    target_lora_path_to_bake = None
    if selected_lora_display_name != "None" and selected_lora_display_name is not None:
        target_lora_path_to_bake = lora_path_mapping_g.get(selected_lora_display_name)
        if not target_lora_path_to_bake:
             # Check legacy names if new mapping fails (for examples primarily)
            legacy_map = {
                'Makoto Shinkai Style': os.path.join(LORAS_DIR, "Makoto_Shinkai_style.safetensors"),
                'Ghibli Style': os.path.join(LORAS_DIR, "ghibli_style.safetensors"),
                'Ghibli Anime Art Style': os.path.join(LORAS_DIR, "Ghibli_Anime_Art_Style.safetensors")
            }
            target_lora_path_to_bake = legacy_map.get(selected_lora_display_name)


    if target_lora_path_to_bake != current_baked_lora_path:
        gr.Info(f"LoRA selection changed (or first load with LoRA). Target: {selected_lora_display_name if selected_lora_display_name else 'None'}. Re-initializing pipeline...")
        print(f"LoRA change detected. Old: {current_baked_lora_path}, New Target: {target_lora_path_to_bake}")
        initialize_pipeline_and_models(lora_file_path_to_bake=target_lora_path_to_bake)
        # current_baked_lora_path is updated inside initialize_pipeline_and_models
        if pipe is None: # If re-initialization failed
            gr.Error("Critical Error: Pipeline re-initialization failed during LoRA change!")
            return [], seed
        gr.Info("Pipeline re-initialized with new LoRA (or base model).")
    
    print(f"Starting generation loop: {num_generations} image(s). Current baked LoRA: {current_baked_lora_path}")
    try:
        processed_image = remove_bkg(input_image)
    except Exception as e:
        gr.Error(f"Failed to process input image for background removal: {e}")
        return [], seed

    all_generated_images = []
    current_seed = int(seed)

    for i in range(int(num_generations)):
        iteration_seed = random.randint(0, MAX_SEED) if randomize_seed else current_seed
        gr.Info(f"Generating image {i+1}/{int(num_generations)} with seed: {iteration_seed}")
        print(f"--- Generation {i+1}/{int(num_generations)} --- Seed: {iteration_seed} ---")

        generator = torch.Generator(device=device_str).manual_seed(iteration_seed)

        # Add trigger phrase for baked LoRA if applicable
        final_prompt = prompt_text
        if current_baked_lora_path and selected_lora_display_name != "None":
            # Derive trigger from display name (simple version)
            # More robust would be to store triggers with LoRAs if they differ from name
            trigger = selected_lora_display_name.lower()
            if trigger not in final_prompt.lower():
                final_prompt = f"{prompt_text}, {trigger}"
                print(f"Added trigger phrase '{trigger}' to prompt for baked LoRA.")
        
        common_args = dict(
            prompt=final_prompt,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=guidance_scale,
            width=1024, height=1024, # Assuming fixed size for now
            subject_image=processed_image,
            subject_scale=scale,
            generator=generator,
        )

        images_batch = []
        try:
            print("Generating image with current pipeline state (LoRA baked in)...")
            # The `with_style_lora` method is NOT used. LoRA is part of the `pipe`'s weights.

            if args.compile_model and torch.cuda.is_available(): # CUDAGraphs are CUDA specific
                torch.compiler.cudagraph_mark_step_begin()

            images_batch = pipe(**common_args).images
            print(f"Iteration {i+1} complete. Generated {len(images_batch)} image(s).")

            if isinstance(images_batch, list):
                for idx, img in enumerate(images_batch):
                    saved_path = save_image(img, iteration_seed, idx, final_prompt)
                    if saved_path: all_generated_images.append(img)
            elif isinstance(images_batch, Image.Image):
                 saved_path = save_image(images_batch, iteration_seed, 0, final_prompt)
                 if saved_path: all_generated_images.append(images_batch)

        except Exception as e:
            print(f"!!! Error during pipeline execution (Iteration {i+1}): {e}")
            import traceback
            traceback.print_exc()
            gr.Warning(f"Image generation failed on iteration {i+1}. Check logs.")

        if not randomize_seed: current_seed += 1
            
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
        # time.sleep(0.1) # Small delay might help system catch up, optional

    print(f"--- Generation loop finished. Total images: {len(all_generated_images)} ---")
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    next_seed_to_return = current_seed if not randomize_seed else seed
    return all_generated_images, next_seed_to_return

def run_for_examples(source_image_path, prompt, scale, style_mode):
    print(f"Running example: {source_image_path}, Prompt: '{prompt}', Scale: {scale}, Style: {style_mode}")
    try:
        input_image_pil = Image.open(source_image_path)
    except FileNotFoundError:
        gr.Warning(f"Example image {source_image_path} not found!")
        return [], 12345
    except Exception as e:
        gr.Error(f"Could not load example image {source_image_path}: {e}")
        return [], 12345

    example_seed = 12345
    fixed_guidance = 3.5
    fixed_steps = args.default_steps if args else 40 # Use app's default steps
    num_generations = 1
    
    generated_images, _ = run_generation_loop(
        input_image=input_image_pil, prompt_text=prompt, scale=scale,
        guidance_scale=fixed_guidance, num_inference_steps=fixed_steps,
        seed=example_seed, randomize_seed=False, # Examples use fixed seed
        style_mode=style_mode, num_generations=num_generations,
    )
    print("Example run finished.")
    return generated_images # Examples only update gallery

css = "footer {visibility: hidden;} .gr-image { min-width: 250px !important; } .gr-gallery { min-height: 400px !important; }"

# Determine initial default LoRA value for the UI
initial_default_lora_value = "None"
if available_loras_g: # Check if list is not empty
    has_none_choice_initial = any(val == "None" for _, val in available_loras_g)
    if not has_none_choice_initial: # If "None" is not among choices
        initial_default_lora_value = available_loras_g[0][1] # Pick first available LoRA's value
    # If "None" is a choice, initial_default_lora_value remains "None"
elif not available_loras_g: # If no LoRAs found at all (empty list including no "None")
    initial_default_lora_value = None # Or handle as appropriate, gr.Dropdown might error with None value if choices are also empty

with gr.Blocks(css=css, theme=gr.themes.Soft()) as block:
    gr.Markdown("# InstantCharacter SECourses Improved App V4 - https://www.patreon.com/posts/126995127")
    with gr.Row():
        with gr.Column():
            image_pil = gr.Image(label="Source Character Image", type='pil', height=768, width=768)
            generate_button = gr.Button("Generate Image", variant="primary", scale=1)
            prompt_input = gr.Textbox(label="Prompt", info="Describe scene and action.", value="a character is riding a bike in snow")
            with gr.Row():
                scale_slider = gr.Slider(minimum=0.0, maximum=1.5, step=0.01, value=1.0, label="Character Scale", info="Adherence to source.", scale=1)
                style_dropdown = gr.Dropdown(label='Artistic Style', choices=available_loras_g, value=initial_default_lora_value, info="Select LoRA style.", scale=1)
            with gr.Row():
                refresh_loras_button = gr.Button("🔄 Refresh LoRAs", scale=1)
                open_loras_button = gr.Button("📁 Open LoRAs Folder", scale=1)
            num_generations_input = gr.Number(label="Number of Generations", value=1, minimum=1, step=1, info="How many images.")

        with gr.Column():
            gallery_output = gr.Gallery(label="Generated Image(s)", object_fit="contain", columns=2, preview=True, height=768)
            open_folder_button = gr.Button("Open Outputs Folder")
            with gr.Accordion("Advanced Options", open=True):
                 with gr.Row():
                    cfg_slider = gr.Slider(minimum=1.0, maximum=10.0, step=0.1, value=3.5, label="Guidance Scale (CFG)", info="Prompt strength.", scale=1)
                    steps_slider = gr.Slider(minimum=5, maximum=100, step=1, value=40, label="Inference Steps", info="More steps = more detail.", scale=1) # Increased max steps
                 with gr.Row():
                     seed_slider = gr.Slider(minimum=0, maximum=MAX_SEED, value=random.randint(0, MAX_SEED), step=1, label="Seed", info="Set for reproducibility.", scale=3)
                     randomize_checkbox = gr.Checkbox(label="Randomize seed", value=True, scale=1)

    example_list = get_example()
    if example_list:
        gr.Examples(examples=example_list, inputs=[image_pil, prompt_input, scale_slider, style_dropdown],
                    outputs=[gallery_output], fn=run_for_examples, cache_examples=False, label="Examples")
    else: gr.Markdown("_(No example images found in 'assets' folder)_")

    generate_button.click(fn=run_generation_loop,
        inputs=[image_pil, prompt_input, scale_slider, cfg_slider, steps_slider, seed_slider, randomize_checkbox, style_dropdown, num_generations_input],
        outputs=[gallery_output, seed_slider], show_progress="full")
    open_folder_button.click(fn=open_folder_outputs, inputs=[], outputs=[])
    open_loras_button.click(fn=open_loras_folder, inputs=[], outputs=[])
    refresh_loras_button.click(fn=refresh_loras, inputs=[], outputs=[style_dropdown])

print("Gradio interface built.")

# --- Launch Application ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='InstantCharacter Gradio App with Offloading and Compilation.')
    parser.add_argument('--highvram', action='store_true', help='Disable CPU offloading (requires high VRAM).')
    parser.add_argument('--compile_model', action='store_true', default=False, help='Enable torch.compile (default: True).') # Default True
    parser.add_argument('--no-compile', action='store_false', dest='compile_model', help='Disable torch.compile.')
    parser.add_argument('--share', action='store_true', help='Enable Gradio sharing link.')
    parser.add_argument('--host', type=str, default=None, help='Host name (e.g., 0.0.0.0).')
    parser.add_argument('--port', type=int, default=None, help='Port number.')
    parser.add_argument('--default_steps', type=int, default=40, help='Default inference steps for UI and examples.')

    args = parser.parse_args() # Populate global args

    print(f"--- App Configuration ---")
    print(f"Device: {device_str}, Dtype: {dtype_torch}")
    print(f"High VRAM mode: {'Enabled (No Offload)' if args.highvram else 'Disabled (Offload Active)'}")
    print(f"Torch Compile: {'Enabled' if args.compile_model else 'Disabled'}")
    print(f"Default Inference Steps: {args.default_steps}")
    if args.compile_model: # Set TorchDynamo configs if compiling
        print("Setting torch._dynamo.config for compilation...")
        os.makedirs(TORCH_CACHE_DIR, exist_ok=True) # Ensure cache directory exists
        print(f"Setting TORCH_COMPILE_CACHE_DIR to: {TORCH_CACHE_DIR}")
        os.environ['TORCH_COMPILE_CACHE_DIR'] = TORCH_CACHE_DIR # Set cache directory using environment variable
        torch._dynamo.config.cache_size_limit = 1024 # As per demo
        torch.set_float32_matmul_precision("high") # As per demo
        torch._dynamo.config.capture_scalar_outputs = True # As per demo
        torch._dynamo.config.capture_dynamic_output_shape_ops = True # As per demo
        torch._dynamo.reset() # Good practice before first compilation

    # Initial setup directories (already done globally now, but harmless to repeat os.makedirs)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LORAS_DIR, exist_ok=True)
    os.makedirs("assets", exist_ok=True) # Ensure assets dir for examples

    # LoRA downloads are now done before UI definition.
    # The available_loras_g is already populated.
    # The style_dropdown is initialized with these choices and a determined default.
    # So, the following lines that try to manually set choices/value can be removed.
    # import shutil # Already imported for startup downloads
    # for filename, repo_id in lora_files_to_download.items():
    #     local_path = os.path.join(LORAS_DIR, filename)
    #     if not os.path.exists(local_path):
    #         print(f"Downloading {filename} from {repo_id} to {LORAS_DIR}...")
    #         try:
    #             downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename)
    #             shutil.copy(downloaded_path, local_path)
    #         except Exception as e:
    #             print(f"Failed to download LoRA {filename}: {e}")
    
    # available_loras_g, lora_path_mapping_g = get_available_loras() # This would re-scan, globals are already set
    # if style_dropdown is not None: # This check is not very useful here as it's always defined before launch.
    #     # Determine default_lora based on the new format of available_loras_g (list of tuples)
    #     default_lora_val = "None" # Default to the value "None"
    #     has_none_choice_main = any(val == "None" for _, val in available_loras_g)
    #     if not has_none_choice_main and available_loras_g:
    #         default_lora_val = available_loras_g[0][1] # Get value from first tuple
    #     elif not available_loras_g:
    #         default_lora_val = None
        # These direct assignments are not the way to update Gradio UI components post-definition.
        # The component should be initialized correctly, or updated via a callback (like refresh_loras_button does).
        # style_dropdown.choices = available_loras_g # Incorrect way to update
        # style_dropdown.value = default_lora_val   # Incorrect way to update


    setup_global_paths()      # Download/resolve paths for main model components
    load_matting_model()      # Load BiRefNet
    
    gr.Info("Initializing main pipeline (base model)... This may take a moment, especially with compilation.")
    initialize_pipeline_and_models() # Initial load of base pipeline (no LoRA baked yet)
    if pipe is None:
        gr.Error("CRITICAL: Main pipeline failed to initialize on startup. The application cannot run.")
        print("CRITICAL: Main pipeline failed to initialize on startup. Exiting.")
        exit()
    gr.Info("Main pipeline initialized. Ready for generation.")


    print("Launching Gradio app...")
    block.queue(max_size=10) # Max 10 concurrent requests
    block.launch(inbrowser=True, share=args.share)

# --- END OF REVISED FILE secourses_app.py ---