## Proyecto de Programación Lineal MAT-24410

- Danya Carolina Gómez Cantú 198618
- Luis Fernando Rodríguez Retama 208047
- Diego Nieto Pizano 207606

Este proyecto de Programación Lineal implementa **Separación Lineal** para clasificar dos grupos de observaciones A y B en Rⁿ. Partimos del modelo que busca un **hiperplano**  

    H(w, β) = { x ∈ Rⁿ | wTx = β }

que, si es posible, **separe** ambos conjuntos; si no lo es, **minimiza** las violaciones mediante **variables de holgura**.  

Se resuelven **primal** y **dual** con `scipy.optimize.linprog`, y se reportan iteraciones, tiempo de CPU, valor óptimo, verificación KKT y gráficas de:

    A w + y   y   B w - z

Como primer dataset usamos **Breast Cancer Wisconsin (Diagnostic)** del UCI ML Repository (569 instancias, 30 variables reales y etiqueta Diagnosis ∈ {M, B}).  

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
