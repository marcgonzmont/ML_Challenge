####################################################
# Cargar el modelo del clasificador aprendido      #
####################################################
test_filename  = "test.csv"
import pickle
model_filename = "Alfredo_SKlearn.model" #<- Poned aqui el nombre de vuestro modelo!!
modelP2 = pickle.load(open(model_filename, 'rb'))

''' ===================================================================================
De aquí en adelante debes escribir tu código para probar lo bueno que es tu modelo
Recuerda que:
 1. El fichero .csv con los datos de test estará en la misma carpeta que éste fichero.
 2. En este código debes incluir todo el prepocesado de datos que sea necesario.
    El fichero de test tiene exactamente las mismas características que el fichero de 
     entrenamiento que habeis recibido, excepto, quizas, el número de ejemplos.
 3. Para medir el rendimiento de tu modelo debes presentar por pantalla:
    -- la matriz de confusión normalizada
    -- la suma de los elementos de su diagonal principal
______________________________________________________________________________________'''

# Código para leer el conjunto de test

# Código para preprocesar el conjunto de test

# Código para estimar las etiquetas del conjunto de test

# Código para obtener la matriz de confusión

# Código para sumar los elementos de su diagonal principal

print("\n=== RESULTADO DE Alfredo & SKlearn ====") #<- Poned aqui vuestros nombres!!
# Código para mostrar por pantalla los resultados
