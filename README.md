# Fenómenos del Caribe — datos

Repositorio **público de datos** generados por robots de GitHub Actions a
partir de fuentes abiertas oficiales. El código del sitio vive en un repo
privado; aquí solo hay datos derivados y los scripts que los producen.

| Carpeta | Contenido | Fuente | Cadencia |
|---|---|---|---|
| `goes/` | Satélite GOES-19 (IR canal 13) reproyectado a la región del Caribe, WebP con transparencia + `meta.json` | AWS Open Data (`noaa-goes19`) | ~10 min |

La app lee estos archivos vía `raw.githubusercontent.com`. La historia del
repo se re-crea en un solo commit en cada publicación para que su tamaño
se mantenga constante.

Datos de NOAA (dominio público). Este repositorio no contiene código de la
aplicación.
