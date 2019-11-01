# Contenidos de esta carpeta
* El fichero *ks_flux_correct.edp* está diseñado para resolver numéricamente el sistema de Keller-Segel con un método de corrección de flujo. 
* Como parámetro, se le puede pasar el tamaño de la malla (nx) y el número de iteraciones (nt).
* Por defecto, se  ejecuta una sola iteración y se utiliza una malla grosera (nx=5) para que el tamaño de las matrices no sea demasiado grande (36 filas y columnas)
*  programa .edp graba las distintas matrices intermedias en respectivos ficheros, que están en [formato COO](https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)) (formato usado por FreeFem++)
* El *Notebook* de Jupyter *inspect_matrices.ipynb* está diseñado para abrir estos ficheros e inspeccionar la estructura de las matrices
