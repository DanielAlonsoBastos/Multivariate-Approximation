# LIBRERIAS
import numpy as np
import os
import matplotlib.pyplot as plt

# FUNCIONES
def readData(path,delimiter='tab'):
    '''
    Lee un archivo csv o txt de la ruta especificada
    
    Inputs
    ----------
    path: STRING, ruta del archivo que se desea procesar
    delimiter: STRING, tipo de separador por coma (,) o tabulador (tab)
    
    Returns
    -------
    data : ARRAY, arreglo con los datos del archivo
    '''
    try:
        if delimiter == 'tab':
            data = np.genfromtxt(path,delimiter='\t')
        elif delimiter == ',':
            data = np.genfromtxt(path,delimiter=',')     
        return data
    except:
        print('El separador seleccionado no es compatible')
        
def getXY(data):
    '''
    De un arreglo de datos, ordena los datos en X y separa las variables X y Y
    en distintos arreglos

    Inputs
    ----------
    data: ARRAY, arreglo con los datos del archivo

    Returns
    -------
    n : INT, numero de elementos del arreglo de la variable independiente X
    x : ARRAY, vector con los datos de la variable independiente X
    y : ARRAY, vector con los datos de la variable dependiente Y
    x_min: FLOAT, valor minimo de la variable independiente X
    x_max: FLOAT, valor maximo de la variable independiente X
    '''
    # ordenar variable independiente X, de menor a mayor
    data = data[np.argsort(data[:,0])]
    # obtener cantidad de datos
    n = len(data)
    # inicializar arreglos
    x = np.zeros(n)
    y = np.zeros(n)
    # asignar valores
    for i in range(n):
        x[i] = data[i,0]
        y[i] = data[i,1]
    
    # calcular limite inferior y superior de X
    x_min = x[0]
    x_max = x[-1]
    return n,x,y, x_min, x_max

def SpCoef(n,x,y):
    '''
    Calcula el valor de la segunda derivada de la spline.

    Inputs
    ----------
    n : INT, numero de elementos del arreglo de la variable independiente X
    x : ARRAY, vector con los datos de la variable independiente X
    y : ARRAY, vector con los datos de la variable dependiente Y
    
    Parameters
    ----------
    sigma, tau : ARRAY, arreglos auxiliares de tamanio n

    Returns
    -------
    s : ARRAY, arreglo con el valor de la segunda derivada en cada punto
    '''
    # inicializar arreglos
    sigma = np.zeros(n)
    tau = np.zeros(n)
    s = np.zeros(n) # arreglo con el valor de las segundas derivadas (s[0] = s[n-1] = 0)
    
    # calcular segunda derivada [1,n-1]
    for i in range(1,n-1):
        hi_1 = x[i] - x[i-1]
        hi = x[i+1] - x[i]
        k = (hi_1/hi)*(sigma[i]+2)+2
        sigma[i+1] = -1/k
        d = (6/hi)*((y[i+1]-y[i])/hi - (y[i]-y[i-1])/hi_1)
        tau[i+1] = (d-hi_1*tau[i]/hi)/k
        
    # calcular hacia atras los valores de la segunda derivada
    for i in reversed(range(1,n-1)):
        s[i]=sigma[i+1]*s[i+1]+tau[i+1]

    return s

def Spline(n,x,y,s,alfa):
    '''
    Interpolacion de un valor alfa del spline

    Inputs
    ----------
    n : INT, numero de elementos del arreglo de la variable independiente X
    x : ARRAY, vector con los datos de la variable independiente X
    y : ARRAY, vector con los datos de la variable dependiente Y
    s : ARRAY, arreglo con el valor de la segunda derivada en cada punto
    alfa : FLOAT, valor de la variable independiente a interpolar

    Returns
    -------
    beta : FLOAT, valor de la variable dependiente
    '''
    # obtener intervalo donde se encuentra alfa
    for i in range(n):
        if alfa<=x[i]:
            break
    i=i-1
    a=x[i+1]-alfa
    b=alfa-x[i]
    hi=x[i+1]-x[i]
    beta = a*s[i]*(a*a/hi-hi)/6+b*s[i+1]*(b*b/hi-hi)/6+(a*y[i]+b*y[i+1])/hi
    return beta

def calculateSpline(x_min,x_max,m,n,x,y,s):
    '''
    Calcular la beta para una cantidad m de puntos en el intervalo [x_min, x_max]

    Inputs
    ----------
    x_min: FLOAT, valor minimo de la variable independiente X
    x_max: FLOAT, valor maximo de la variable independiente X
    m : INT, cantidad de puntos a interpolar
    n : INT, numero de elementos del arreglo de la variable independiente X
    x : ARRAY, vector con los datos de la variable independiente X
    y : ARRAY, vector con los datos de la variable dependiente Y
    s : ARRAY, arreglo con el valor de la segunda derivada en cada punto

    Returns
    -------
    x_spline : ARRAY, vector con los m puntos de la variable independiente X
    y_spline : ARRAY, vector con las m interpolaciones de la variable dependiente Y
    '''
    # arreglo con m puntos en el intervalo [x_min,x_max]
    x_spline = np.linspace(x_min,x_max,m)
    # inicializar arreglo que guardara las betas resultantes
    y_spline = np.zeros(m)
    # calcular betas
    for i in range(m):
        y_spline[i] = Spline(n,x,y,s,x_spline[i])
        
    return x_spline, y_spline

def plotSpline(x,y,x_spline,y_spline,path):
    '''
    Crear una grafica con los puntos originales y los del spline
    
    Inputs
    ----------
    x : ARRAY, vector con los datos de la variable independiente X
    y : ARRAY, vector con los datos de la variable dependiente Y
    x_spline : ARRAY, vector con los m puntos de la variable independiente X
    y_spline : ARRAY, vector con las m interpolaciones de la variable dependiente Y
    path: STRING, ruta del archivo que se desea procesar

    Returns
    -------
    None.

    '''
    # obtener nombre del archivo
    file_name = os.path.basename(path).split('.')[0]
    
    plt.figure(figsize=(12, 8))
    plt.plot(x, y, 'o', label='data', color = 'red')
    plt.plot(x, y, '-', label='linear')
    plt.plot(x_spline,y_spline,  '-o', markersize=3, label="S", color='green')
    plt.legend(loc='upper right', ncol=2)
    plt.title('Natural Spline')
    plt.savefig(file_name + '_resultados.png')
    print('La grafica se ha guardado como: ', file_name + '_resultados.png')
    plt.show()
    
def saveResults(path,m,x_spline,y_spline):
    '''
    Guardar los resultados en un archivo txt o csv, segun sea el caso

    Inputs
    ----------
    path: STRING, ruta del archivo que se desea procesar
    m : INT, numero de elementos del arreglo de la variable independiente X
    x_spline : ARRAY, vector con los m puntos de la variable independiente X
    y_spline : ARRAY, vector con las m interpolaciones de la variable dependiente Y
    
    Parameters
    ----------
    file_name: STRING, nombre del archivo
    extension : STRING, extension del archivo (csv o txt)
    data_spline : ARRAY, arreglo con los valores de X y Y resultantes

    Returns
    -------
    None.

    '''
    # obtener nombre y extension del archivo
    file_name = os.path.basename(path).split('.')[0]
    extension = os.path.basename(path).split('.')[-1]
    # inicializar arreglo
    data_spline = np.zeros([m,2])
    # colocar valores
    for i in range(m):
        data_spline[i,0] = x_spline[i]
        data_spline[i,1] = y_spline[i]
        
    # guardar resultados
    if extension == 'txt':
        np.savetxt(file_name + '_resultados.txt', data_spline, delimiter='\t',fmt='%10.15f')
        print('Los resultados se encuentran en: ', file_name + '_resultados.txt')
    if extension == 'csv':
        np.savetxt(file_name + '_resultados.csv', data_spline, delimiter=',',fmt='%10.15f')
        print('Los resultados se encuentran en: ', file_name + '_resultados.txt')
    
       
def main(path,delimiter,m):
    '''
    Funcion principal que ejecuta el programa
    1. Leer el archivo
    2. Ordenar de menor a mayor la variable independiente X
    3. Separar en vectores las variables
    4. Calcular el numero de datos en la variable (n)
    5. Obtener el limite inferior y superior de la variable independiente X
    6. Calcular el vector de segundas derivadas (s)
    7. Interpolar los m puntos del spline
    8. Graficar
    9. Guardar resultados

    Inputs
    ----------
    path: STRING, ruta del archivo que se desea procesar
    delimiter: STRING, tipo de separador por coma (,) o tabulador (tab)
    m: INT, cantidad de puntos a interpolar

    Returns
    -------
    None.

    '''
    # leer archivo
    data = readData(path,delimiter)
    # obtener valores de n,x,y,x_min,x_max
    n,x,y, x_min, x_max = getXY(data)
    # calcular vector de segundas derivadas
    s = SpCoef(n,x,y)
    # interpolar los valores de alfa para los m puntos en un rango
    x_spline, y_spline = calculateSpline(x_min,x_max,m,n,x,y,s)
    # graficar
    plotSpline(x,y,x_spline,y_spline,path)
    # guardar resultados
    saveResults(path,m,x_spline,y_spline)
    
# MAIN
## input del usuario
path = input('Ingresa la ruta del archivo: ')
delimiter = input('Ingresa el tipo de separador (, o tab):')
try:
    m = int(input('Ingresa el numero m de datos que deseas: '))
except:
    print('El valor tiene que ser un numero entero')

## ejecutar funcion principal
main(path,delimiter,m)