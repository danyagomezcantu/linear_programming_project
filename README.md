# Proyecto de Programación Lineal

Este proyecto de Programación Lineal implementa Separación Lineal para clasificar dos grupos de observaciones A y B en Rn.
Partimos del modelo que busca un hiperplano 𝐻(𝑤,𝛽)={𝑥∈𝑅𝑛∣𝑤⊤x=β} que, si es posible, separe ambos conjuntos.

Se resuelven primal y dual con scipy.optimize.linprog, y se reportan iteraciones, tiempo de CPU, valor óptimo, verificación KKT y gráficas de Aw+y y Bw−z. 

Como primer dataset usamos Breast Cancer Wisconsin (Diagnostic) del UC Irvine ML Repository (569 instancias, 30 variables reales y etiqueta Diagnosis ∈{M,B}). La carga se realiza con ucimlrepo.

### Install the ucimlrepo package 
pip install ucimlrepo

### Import the dataset into your code 

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