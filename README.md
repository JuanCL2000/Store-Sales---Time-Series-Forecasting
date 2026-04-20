# Store Sales - Time Series Forecasting

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-Forecasting-green.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-orange.svg)

## Resumen del Proyecto
Este proyecto aborda el desafío de predecir las ventas unitarias de miles de artículos en diferentes tiendas de **Corporación Favorita**, una gran cadena de supermercados con sede en Ecuador. 

Los pronósticos precisos son cruciales en la industria minorista: sobrestimar la demanda resulta en exceso de inventario y desperdicio de productos perecederos, mientras que subestimarla provoca desabastecimiento, pérdida de ingresos y clientes insatisfechos. 

**Objetivo:** Construir un modelo de Machine Learning para predecir las ventas diarias (variable `sales`) durante un periodo de 16 días, minimizando el Error Cuadrático Medio Logarítmico (RMSLE).

## Los Datos
> **Nota:** Por buenas prácticas de desarrollo y límites de tamaño de Git, los datasets originales no están incluidos en este repositorio.

Los datos pueden ser descargados desde la competencia oficial: [Kaggle: Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data).

El conjunto de datos incluye información detallada sobre:
* **Fechas y tiendas:** Histórico de ventas por tienda y familia de productos.
* **Promociones:** Artículos en descuento en fechas específicas.
* **Factores externos:** Precios diarios del petróleo (fundamental para la economía ecuatoriana) y días festivos/eventos locales y nacionales.

## Enfoque Metodológico (Analytic Approach)
El desarrollo del proyecto siguió el flujo de trabajo estándar de ciencia de datos:

1. **Análisis Exploratorio de Datos (EDA):** Identificación de tendencias, estacionalidad semanal/anual y el impacto de variables exógenas como el precio del petróleo y los días festivos.
2. **Ingeniería de Características (Feature Engineering):** * Extracción de componentes temporales (día de la semana, mes, año).
   * Creación de variables rezagadas (lags) y medias móviles para capturar la inercia de las ventas.
3. **Modelado:** Implementación de **LightGBM** (`LGBMRegressor`), un algoritmo de Gradient Boosting altamente eficiente y robusto para series temporales de gran volumen.
4. **Optimización y Post-procesamiento (Técnicas Clave):**
   * **Transformación de la Variable Objetivo:** Se entrenó el modelo utilizando el logaritmo de las ventas para suavizar distribuciones sesgadas, revirtiendo el cálculo (`np.expm1`) al generar la predicción final.
   * **Restricción de Negocio (Clipping):** Se aplicó `np.clip(predicciones, 0, None)` para forzar un límite inferior de cero, asegurando que el modelo no pronosticara ventas negativas (lo cual es matemáticamente posible, pero lógicamente incorrecto en retail).
   * **Ajuste de Hiperparámetros:** Tuning avanzado de LightGBM (`n_estimators=2000`, `num_leaves=127`) para maximizar la profundidad del aprendizaje previniendo el sobreajuste.

## Resultados y Evaluación
El modelo fue evaluado utilizando la métrica oficial de la competencia, **RMSLE** (Root Mean Squared Logarithmic Error). Esta métrica penaliza los errores relativos y es ideal cuando hay un gran rango en los valores objetivo.

* **Score RMSLE de Validación:** `0.48`

## Cómo reproducir este proyecto

Si deseas ejecutar este proyecto localmente, sigue estos pasos:

1. Clona este repositorio:
   ```bash
   git clone [https://github.com/JuanCL2000/Store-Sales---Time-Series-Forecasting.git](https://github.com/JuanCL2000/Store-Sales---Time-Series-Forecasting.git)
   cd Store-Sales---Time-Series-Forecasting

2. Instala las dependencias necesarias:

pip install -r requirements.txt

3. Descarga los datos desde Kaggle y extrae los archivos .csv en una carpeta llamada data/raw/ en la raíz del proyecto.

4. Ejecuta la libreta de Jupyter:

jupyter notebook notebooks/[01_data_loading_and_merging].ipynb

### Autor

Juan Cuellar - [www.linkedin.com/in/juan-cuellar-lugo-3a55b3374]

GitHub: JuanCL2000
