# INSTRUCCIONES.md

Este archivo gobierna a cualquier agente de código (Muse Code, Claude Code u otro) que trabaje en este repositorio. Léelo completo antes de cada prompt de PROMPT_CHAIN.md. Si un prompt contradice este archivo, gana este archivo y la contradicción se anota en PROGRESS.md.

## 1. Qué es el proyecto

**EgressTwin** (nombre de trabajo): un agente de IA verificable que convierte la imagen de un espacio (plano de un edificio o foto aérea de un espacio abierto) en un gemelo digital de evacuación, lo simula en paralelo y solo recomienda cambios que puede demostrar con simulaciones.

Pipeline del producto:

1. **Percepción**: imagen -> `Layout` (zona caminable, paredes, obstáculos, salidas, escala) usando Muse Spark 1.3 (comprensión de imagen y salida estructurada) y SAM 3.1 (segmentación), con respaldo OpenCV y corrección manual del usuario.
2. **Simulación**: modelo de fuerza social con campo de piso geodésico, kernels Numba en paralelo.
3. **Barrido distribuido**: escenarios x réplicas estocásticas con mpi4py o multiprocessing, con intervalos de confianza.
4. **Agente**: Muse Spark 1.3 con tool calling propone modificaciones, las simula, hace pruebas de robustez N-1 (cada salida bloqueada) y entrega un reporte estructurado.
5. **Verificador determinista**: código (no LLM) que contrasta cada afirmación del reporte contra los resultados guardados.
6. **Web app**: FastAPI + una página HTML/JS con canvas.

Contexto: Meta Global AI Developer Hackathon, indie track. Construcción del 27 oct al 4 nov de 2026. Entrega el 4 nov a las 9:00 PM ET.

## 2. Reglas del hackathon (no negociables)

- **Solo trabajo nuevo.** Todo el código se escribe a partir del 27 oct de 2026. Está prohibido copiar código del prototipo C++ de la materia de Computación Paralela y Distribuida. Si se reutiliza cualquier fragmento previo, se declara en NEW_WORK.md con su origen.
- **Repo público.** Nunca se commitean secretos. `.env` va en `.gitignore`; solo existe `.env.example`.
- **Assets con derechos.** Solo imágenes y videos propios, generados por el proyecto o con licencia que permita su uso. Cada asset se registra en `data/ASSETS.md` con origen y licencia.
- **Encuadre.** El producto es una herramienta de apoyo a la decisión, no una certificación de seguridad. Nada de consejo médico. Nada de escenarios de guerra, conflicto o política.
- **Historial legible.** Commits pequeños y descriptivos. El historial es evidencia de que el trabajo se hizo durante el evento.

## 3. Stack

- Python 3.11. Gestor: `uv` recomendado, `pip` + `venv` válido.
- Núcleo: `numpy`, `numba`, `scipy`, `mpi4py`, `pydantic` v2, `pydantic-settings`.
- Percepción: `opencv-python-headless`, `pillow`, `openai` (SDK apuntando a Meta Model API), parser oficial de SAM para Python (verificar nombre exacto y versión en PyPI; la documentación lo cita como `meta-sam-parser`, versión 0.0.6 o superior si se pide confianza).
- Web: `fastapi`, `uvicorn`, `sse-starlette` o equivalente. Frontend: un solo `index.html` con JS vanilla y canvas, sin build step.
- Calidad: `pytest`, `ruff`, `matplotlib` para figuras.
- Hardware objetivo de desarrollo: Ryzen 3 3300X (4 núcleos, 8 hilos), 16 GB RAM, GPU AMD RX 570. **No hay CUDA**: todo el paralelismo es CPU (hilos Numba + procesos MPI).
- En Windows, `mpi4py` requiere MS-MPI instalado. Si da problemas, usar WSL2. Los scripts de tareas se escriben en Python (`scripts/`) para que funcionen en Windows y Linux; no depender de `make`.

## 4. Estructura del repositorio

```
egresstwin/
  README.md            # en inglés, para jueces
  NEW_WORK.md          # declaración de trabajo nuevo
  INSTRUCCIONES.md     # este archivo
  PROMPT_CHAIN.md      # cadena de prompts
  PROGRESS.md          # bitácora que el agente actualiza al final de cada prompt
  LICENSE              # MIT
  pyproject.toml
  .env.example
  data/
    ASSETS.md          # origen y licencia de cada imagen o video
    samples/           # solo assets con derechos
  src/egresstwin/
    config.py
    meta/              # todo acceso a Meta Model API pasa por aquí
      client.py        # SDK OpenAI con base_url de Meta, reintentos, caché, mock
      costs.py         # tabla de precios y tope de gasto
      spark.py         # helpers de Muse Spark (visión, salida estructurada)
      sam.py           # helpers de SAM 3.1 (concepto, cajas, parseo)
      voice.py         # opcional, Muse Voice Transcribe
    layout/
      schema.py        # modelo Pydantic Layout
      raster.py
      validate.py
      edit.py          # correcciones manuales
    perception/
      plan_reader.py   # Spark lee la imagen
      segmenter.py     # SAM segmenta
      cv_fallback.py   # OpenCV para planos de líneas
      pipeline.py
    sim/
      floor_field.py
      wall_field.py
      cells.py         # cell lists
      social_force.py  # kernels Numba
      engine.py
      events.py        # eventos en vivo (opcional)
      metrics.py
      reference.py     # implementación NumPy de referencia para validar
    scenarios/
      spec.py          # ScenarioSpec y DSL de modificaciones
      apply.py
    parallel/
      sweep.py         # backends serial, local, mpi
      replicas.py
      store.py         # almacén de resultados
    agent/
      prompts/system.md
      tools.py
      loop.py
      verifier.py
      trace.py
    validation/        # opcional, video real
    app/
      api.py
      static/index.html
  bench/
  scripts/
  tests/
    fixtures/meta/     # respuestas grabadas para modo mock
  docs/
    figures/
  runs/                # salidas locales, en .gitignore
```

## 5. Convenciones para Meta Model API

- Cliente: SDK `openai` con `base_url="https://api.meta.ai/v1"` y la llave en la variable de entorno `MODEL_API_KEY`.
- IDs de modelo solo en `config.py` (nunca hardcodeados en otros módulos):
  - `META_SPARK_MODEL` por defecto `muse-spark-1.3`.
  - `META_SAM_MODEL` por defecto `sam-3.1`.
  - `META_VOICE_MODEL` por defecto `muse-voice-transcribe-1.0` (opcional).
- Variante barata: `muse-spark-1.3-contributor` cuesta mucho menos, pero Meta usa esos datos para mejorar sus productos. Usarla solo en desarrollo con datos sintéticos o públicos, nunca con video de personas reales.
- Agente: Responses API con `previous_response_id`. `tool_choice` solo admite `"auto"`. El modelo puede devolver varias `function_call` en un turno: ejecutarlas todas y devolver un `function_call_output` por cada `call_id`. Validar siempre los argumentos con Pydantic aunque el esquema sea estricto.
- SAM 3.1:
  - Un concepto por request, como frase nominal corta ("door", "tent"). Sin listas, sin negaciones, sin frases espaciales. Para varios conceptos, varios requests.
  - Los prompts de caja y punto usan `w` y `h` iguales al tamaño real de la imagen en píxeles.
  - Pedir `metadata.include_confidence = "true"` y filtrar por umbral configurable.
  - En video rastrea como máximo 16 objetos por frame. Para contar multitudes se usa modo imagen frame por frame (devuelve todas las instancias).
  - Cero coincidencias es un resultado válido, no un error.
- Toda llamada pasa por `meta/client.py`, que implementa:
  - reintentos con backoff exponencial y timeout,
  - caché en disco por hash del input (imagen + prompt + modelo),
  - registro de tokens, segundos y costo estimado en `runs/costs.jsonl`,
  - tope de gasto `META_BUDGET_USD` (por defecto 60) que lanza una excepción clara al superarse.
- **Modo mock**: con `META_MOCK=1` el cliente responde con fixtures de `tests/fixtures/meta/`. Los tests y la CI nunca llaman a la API real.
- Si la API se comporta distinto a lo descrito aquí, la fuente de verdad es https://dev.meta.ai/docs. Documentar la diferencia en PROGRESS.md.

## 6. Estándares de código

- Código, identificadores, comentarios, logs y README en **inglés** (jueces internacionales). INSTRUCCIONES.md y PROMPT_CHAIN.md quedan en español.
- Sin emojis en código, logs, UI ni documentación.
- Tipado en todas las funciones públicas. Docstrings cortos.
- Simulador en unidades SI (metros, segundos, kilogramos). La escala `meters_per_pixel` es explícita en el `Layout`.
- Kernels Numba: solo arrays NumPy y escalares, sin objetos Python. Cada agente escribe solo su propia fila de fuerzas para evitar condiciones de carrera.
- Semillas explícitas en todo lo estocástico. Mismo `seed` = mismo resultado.
- Funciones puras donde se pueda: `apply_modifications(layout, mods) -> Layout` nunca modifica el original.
- `ruff check` y `pytest` en verde antes de cada commit.
- Nunca inventar números en README, docs ni UI. Si un resultado no existe todavía, se deja `TBD`.

## 7. Flujo de trabajo por prompt

1. Leer INSTRUCCIONES.md, PROGRESS.md y solo el prompt actual de PROMPT_CHAIN.md.
2. No adelantar trabajo de prompts futuros.
3. Implementar, escribir tests y correrlos.
4. Actualizar PROGRESS.md con: fecha, prompt, qué se hizo, decisiones tomadas, problemas abiertos, costo API acumulado (leído de `runs/costs.jsonl`).
5. Commit con el mensaje que indica el prompt.
6. Si un criterio de aceptación no se cumple, no se avanza: se documenta y se pide indicación.

## 8. Definición de terminado del proyecto

- `python scripts/demo.py --mock` corre todo el flujo sin red.
- `python scripts/demo.py` corre el flujo real con la API.
- La demo muestra: imagen -> layout corregible -> simulación base -> agente optimizando en vivo -> recomendación con sello VERIFIED -> prueba N-1 -> benchmarks de paralelismo.
- README con resultados reales y NEW_WORK.md completo.

## 9. Prioridades si falta tiempo

Orden de recorte (lo primero se corta primero):

1. Eventos por voz (P07).
2. Validación con video real (P08).
3. Backend MPI (se mantiene `local` con multiprocessing; documentar).
4. Soporte de foto aérea (se mantiene solo planos).

Nunca se recorta: el verificador, la corrección manual del layout, el modo demo con resultados en caché y la estabilidad de la demo.
