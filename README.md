# HELIX - Generador de Imágenes con Inteligencia Artificial

Script en Python para generar imágenes a partir de texto usando el modelo **Stable Diffusion Turbo** de Stability AI a través de Hugging Face Diffusers.

**Autor:** Frank Erick Joel Diaz Albarracin

## Características
- Detecta automáticamente GPU (CUDA), Apple Silicon (MPS) o CPU
- Modelo muy rápido (1-4 pasos de inferencia)
- Guarda las imágenes en la carpeta `imagenes_generadas/`

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
python helix_ai_image.py
```

Luego escribe tu prompt y espera a que se genere la imagen.

## Notas
- La primera vez que ejecutes el script, el modelo se descargará automáticamente (~2.5 GB).
- En CPU es más lento. Se recomienda GPU NVIDIA o Apple Silicon para mejor rendimiento.
