## Proyecto de Programación Lineal MAT-24410

- Danya Carolina Gómez Cantú 198618  
- Luis Fernando Rodríguez Retama 208047  
- Diego Nieto Pizano 207606  

Este proyecto de Programación Lineal implementa **Separación Lineal** para clasificar dos grupos de observaciones A y B en $\mathbb{R}^n$.  
Partimos del modelo que busca un **hiperplano**

```
H(w, β) = { x ∈ Rⁿ | wᵀx = β }
```

que, si es posible, **separe** ambos conjuntos; y si no lo es, **minimiza** las violaciones mediante **variables de holgura**.  

Se resuelven **primal** y **dual** con `scipy.optimize.linprog`, y se reportan:  
- número de iteraciones,  
- tiempo de CPU,  
- valor óptimo de la función objetivo,  
- validación de condiciones **KKT**,  
- y gráficas de:  

```
A w + y    y    B w - z
```

Como primer dataset se utiliza **Breast Cancer Wisconsin (Diagnostic)** del UCI ML Repository  
(569 instancias, 30 variables reales y etiqueta Diagnosis ∈ {M, B}).  

La carga se realiza con `ucimlrepo` de la siguiente manera:

```python
# Install the ucimlrepo package 
pip install ucimlrepo

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets 
  
# metadata 
print(breast_cancer_wisconsin_diagnostic.metadata) 
  
# variable information 
print(breast_cancer_wisconsin_diagnostic.variables)  
```

---

### Resumen

Este proyecto se desarrolló en el marco del curso de Programación Lineal, con el objetivo de aplicar técnicas de optimización al problema de **separación lineal de conjuntos**. En particular, se buscó encontrar un hiperplano en el espacio $\mathbb{R}^{30}$ capaz de discriminar dos grupos de observaciones —tumores malignos y benignos— a partir del conjunto de datos *Breast Cancer Wisconsin (Diagnostic)*. Este enfoque permite formular la tarea de clasificación como un modelo de **programación lineal**, donde las restricciones describen la posición relativa de los puntos respecto al hiperplano y las variables de holgura cuantifican las violaciones cuando los conjuntos no son perfectamente separables.

Para resolver el problema, se formularon y compararon el modelo **primal** y su correspondiente **dual**, ambos implementados en Python mediante la función `linprog` de la biblioteca `scipy.optimize`, utilizando el método *simplex* clásico. Se midieron variables como el número de iteraciones, el tiempo de ejecución y el valor óptimo de la función objetivo, además de verificarse las condiciones de **Karush–Kuhn–Tucker (KKT)** para garantizar la calidad de la solución. Para facilitar la interpretación geométrica, los resultados se proyectaron a dos dimensiones mediante **Análisis de Componentes Principales (PCA)**, lo que permitió visualizar la frontera lineal que separa ambas clases en un plano que conserva la mayor parte de la varianza del conjunto original.

Los resultados mostraron una coincidencia casi exacta entre el primal y el dual, con una brecha de dualidad del orden de $10^{-14}$ y condiciones KKT satisfechas dentro de tolerancias numéricas estrictas. Esto confirma la optimalidad y estabilidad de la solución obtenida. El modelo logró separar de manera efectiva los dos tipos de tumores, mientras que la proyección PCA evidenció una frontera lineal clara entre ambas clases. En conjunto, el trabajo demuestra cómo la programación lineal puede emplearse exitosamente para resolver un problema de clasificación binaria con interpretación geométrica y validación analítica rigurosa.

---

### Estructura del repositorio

- `lp_separation_simplex.py` — Script principal.  
  Construye y resuelve los modelos **primal** y **dual**, calcula KKT, genera figuras y exporta resultados a JSON.
- `outputs_simplex/` — Carpeta generada automáticamente con:
  - `Aw_plus_y_simplex.png`
  - `Bw_minus_z_simplex.png`
  - `pca_hyperplane_simplex.png`
  - `results_simplex.json`

---

### Requisitos

Instala las dependencias del proyecto:

```bash
pip install numpy pandas matplotlib scipy scikit-learn ucimlrepo
```

> Nota: El método `"simplex"` de `scipy.optimize.linprog` se utiliza porque es requisito del proyecto.  
> Puede mostrar una advertencia de deprecación, pero los resultados son correctos.

---

### Ejecución

Ejecuta el script principal:

```bash
python lp_separation_simplex.py
```

Esto imprimirá un resumen en consola y generará las figuras y el archivo de resultados en `outputs_simplex/`.

---

### Resultados principales

- **Iteraciones (Primal / Dual):** 3492 / 1313  
- **Tiempo de CPU (s):** 20.087 / 6.648  
- **Valor óptimo (f.obj.):** 0.0 / -9.49×10⁻¹⁵  
- **Brecha de dualidad:** 9.49×10⁻¹⁵  
- **Condiciones KKT:** satisfechas dentro de tolerancias numéricas  
  - Complementariedad: −9.49×10⁻¹⁵  
  - Factibilidad primal: 8.72×10⁻⁹  
  - Estacionariedad: 0.0  
  - No negatividad (primal): 0.0  
  - No negatividad (dual slack): −3.14×10⁻¹⁴  

---

### Figuras generadas

- **Aw + y:** verificación de factibilidad de la clase *maligna*.  
- **Bw − z:** verificación de factibilidad de la clase *benigna*.  
- **PCA 2D con hiperplano proyectado:** proyección de los datos y la frontera lineal.

---

### Conclusión

El modelo de separación lineal desarrollado resolvió de manera exacta la separación de los conjuntos maligno y benigno, con resultados numéricamente óptimos y comprobación analítica completa.  
La incorporación de **PCA** permitió visualizar la frontera lineal en dos dimensiones, demostrando la aplicabilidad práctica de la programación lineal para problemas de clasificación y separación.

---
*Instituto Tecnológico Autónomo de México*
**Curso:** Programación Lineal (MAT-24410)
**Periodo:** Otoño 2025  
**Lenguaje:** Python 3.10  
