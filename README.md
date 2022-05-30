# Multivariate-Approximation
Curso: Modelos evolutivos y aprendizaje de máquina\
Imparte: Dr. Angel Fernando Kuri-Morales\
IIMAS, UNAM

En el repositorio se encuentra un PDF donde se tiene una síntesis del curso __Modelos evolutivos y aprendizaje de máquina__, además de los códigos empleados por cada sección. A continuación se describen los pasos para realizar un ambiente virtual y las paqueterías necesarias para ejecutar los programas.

1. Descargar el repositorio del curso y descomprimirlo.
![github](https://user-images.githubusercontent.com/32237029/171016742-5c627376-46f6-466e-8e4c-c68dd7b8c1f0.png)
2. Ejecutar la consola de anaconda: __Anaconda Prompt__
![image](https://user-images.githubusercontent.com/32237029/171029180-5b214611-42c3-414b-854a-e4f3bb01d3f1.png)

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

5. Instalar `pip`
```
conda install pip
```

6. Instalar las paqueterías necesarias:
```
pip install numpy
pip install pandas
pip install matplotlib
pip install plotly
pip install seaborn
pip install numba
pip install -U kaleido
```
7. Instalar Jupyter Notebook y Spyder. Abrir la aplicación de Anaconda, seleccionar el ambiente virtual creado `mva`, e instalar las herramientas deseadas (Jupyter Notebook y Spyder).
![anaconda](https://user-images.githubusercontent.com/32237029/171028450-14a94af9-3e64-406e-bcc6-da9cb98845d7.png)

Con estos pasos se ha realizado un ambiente virtual `mva`, el cual tiene las paqueterías necesarias para ejecutar los códigos.
Nota: Si no se desea, no es necesario crear un ambiente virtual nuevo, solo es necesario instalar las paqueterías descritas en (6).
Nota: En caso de que se haya generado un ambiente virtual nuevo, para ejecutar los programas es necesario activarlo con `conda activate mva` desde la terminal, o se puede seleccionar el ambiente y la aplicación desde el GUI de Anaconda, como se muestra en (7).
