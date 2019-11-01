* El fichero *ks_flux_correct.edp* ejecuta una sola iteración de Keller-Segel con corrección de flujo. 
* Se utiliza una malla grosera (nx=5) para que el tamaño de las matrices no sea demasiado grande (36 filas y columnas)
* Este programa .edp graba las distintas matrices intermedias en respectivos ficheros, que están en [formato COO](https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)) (formato usado por FreeFem++)
* El *Notebook* de Jupyter *inspect_matrices.ipynb* está diseñado para abrir estos ficheros e inspeccionar la estructura de las matrices
