# ============================================================
#  HELIX - Generador de Imágenes con Inteligencia Artificial
#  Autor: Frank Erick Joel Diaz Albarracin
#  Modelo: stabilityai/sd-turbo (Stable Diffusion Turbo)
#  Descripción: Genera imágenes a partir de texto (text-to-image)
#               de forma rápida y local usando Hugging Face Diffusers.
# ============================================================

import torch
from diffusers import AutoPipelineForText2Image
from datetime import datetime
import os


def detectar_dispositivo():
    """Detecta automáticamente el mejor dispositivo disponible."""
    if torch.cuda.is_available():
        return "cuda", torch.float16          # NVIDIA GPU
    elif torch.backends.mps.is_available():
        return "mps", torch.float16           # Apple Silicon (M1/M2/M3...)
    else:
        return "cpu", torch.float32           # CPU (más lento)


def main():
    print("=" * 60)
    print("  HELIX - Generador de Imágenes IA")
    print("  Autor: Frank Erick Joel Diaz Albarracin")
    print("=" * 60)
    print()

    # 1. Detectar dispositivo
    dispositivo, dtype = detectar_dispositivo()
    print(f"[*] Dispositivo detectado: {dispositivo.upper()}")
    print(f"[*] Precisión (dtype): {dtype}")
    print()

    # 2. Cargar el modelo
    print("[*] Cargando modelo stabilityai/sd-turbo ...")
    print("    (La primera vez puede tardar varios minutos mientras descarga ~2.5 GB)")
    
    modelo = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sd-turbo",
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
    )
    modelo = modelo.to(dispositivo)
    print("[✓] Modelo cargado correctamente.\n")

    # 3. Pedir el prompt al usuario
    prompt = input("Escribe el prompt de la imagen: ").strip()
    if not prompt:
        prompt = "a beautiful landscape with mountains and a lake at sunset, highly detailed, cinematic lighting"
        print(f"[!] Prompt vacío. Usando prompt por defecto:\n    {prompt}")

    # Negative prompt (para evitar defectos comunes)
    negative_prompt = (
        "Amputee, Bad anatomy, Bad hands, Badly drawn hands, Cloned body, Cloned face, "
        "Disgusting proportions, Dismembered, Duplicate, Extra fingers, Extra hands, "
        "Extra limbs, Fused fingers, Fused hands, Hideous body, Inadequate scale, "
        "Long neck, Malformed limbs, Missing arms, Missing fingers, Missing hands, "
        "Missing limbs, Multiple heads, Mutated hands, Mutation, Mutations, "
        "Poorly drawn face, Too many fingers, Ugly body, ugly, tiling, "
        "poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, "
        "extra limbs, disfigured, deformed, body out of frame, bad anatomy, "
        "watermark, signature, cut off, low contrast, underexposed, overexposed, "
        "bad art, beginner, amateur, distorted face"
    )

    # 4. Generar la imagen
    print("\n[*] Generando la imagen... (esto puede tardar según tu hardware)")
    
    # Nota importante sobre SD-Turbo:
    # - guidance_scale = 0.0 (el modelo no usa classifier-free guidance)
    # - num_inference_steps = 1 a 4 es suficiente (muy rápido)
    imagen = modelo(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=4,      # 1-4 pasos son ideales para Turbo
        guidance_scale=0.0,         # Obligatorio en 0.0 para SD-Turbo
    ).images[0]

    # 5. Guardar la imagen con nombre único
    carpeta = "imagenes_generadas"
    os.makedirs(carpeta, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"{carpeta}/helix_{timestamp}.png"
    imagen.save(nombre_archivo)

    print(f"\n[✓] ¡Imagen generada y guardada exitosamente!")
    print(f"    Archivo: {nombre_archivo}")
    print("\nGracias por usar HELIX - Frank Erick Joel Diaz Albarracin")


if __name__ == "__main__":
    main()
