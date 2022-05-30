# Multivariate-Approximation
Curso: Modelos evolutivos y aprendizaje de máquina\
Imparte: Dr. Angel Fernando Kuri-Morales\
IIMAS, UNAM

En el repositorio se encuentra un PDF donde se tiene una síntesis del curso __Modelos evolutivos y aprendizaje de máquina__, además de los códigos empleados por cada sección. A continuación se describen los pasos para realizar un ambiente virtual y que los programas puedan ser ejecutados sin problema.

1. Descargar el repositorio del curso y descomprimirlo.
![github](https://user-images.githubusercontent.com/32237029/171016742-5c627376-46f6-466e-8e4c-c68dd7b8c1f0.png)
2. Ejecutar la consola de anaconda: __Anaconda Prompt__
![image](https://user-images.githubusercontent.com/32237029/171017529-9143f9da-b1e1-4c9f-9a68-d97ad8d0ead6.png)
3. Crear un ambiente virtual 
```
conda create -n mva python=3.8.13 
```
Nota: Reemplazar `mva` por el nombre del ambiente virtual. A lo largo del documento se utilizará `mva`, por lo que se tendrá que reemplazar en cada paso por el nombre del ambiente definido.
4. Activar el ambiente virtual
```
conda activate mva
```
Nota: Al activar el ambiente virtual debe aparecer el nombre del mismo entre paréntesis en la consola
![image](https://user-images.githubusercontent.com/32237029/171019707-210278a1-f4bc-4dbf-8068-da48717a3482.png)
5. En la terminal, dirigirse al directorio donde se descargó y descomprimió el repositorio.
```
cd <dir-repositorio>
```
![image](https://user-images.githubusercontent.com/32237029/171020074-ea31e302-f341-42e4-85d0-6f2c77bd678a.png)
7. Instalar `pip`
```
conda install pip
```
9. Instalar todas las paqueterías necesarias
