# import argparse
from pandas import read_csv
from sklearn.metrics import classification_report, confusion_matrix
from myPackage import tools as tl
import pickle

if __name__ == '__main__':
    ####################################################
    # Cargar el modelo del clasificador aprendido      #
    ####################################################
    test_filename = "test.csv"


    model_filename = "Marcos_GM_BGG-RFC.model"  # <- Poned aqui el nombre de vuestro modelo!!
    modelP2 = pickle.load(open(model_filename, 'rb'))
    class_names = ['trees', 'grass', 'soil', 'concrete', 'asphalt', 'buildings', 'cars', 'pools', 'shadows']


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
    te_data = read_csv(test_filename, header=0).values
    data, labels = te_data[:, 1:], te_data[:, 0]

    # Código para preprocesar el conjunto de test
    data = tl.normalize(data)
    # Código para estimar las etiquetas del conjunto de test
    labels_pred = modelP2.predict(data)
    # Código para obtener la matriz de confusión
    cnf_matrix = confusion_matrix(labels, labels_pred)
    tl.np.set_printoptions(precision=3)
    # Código para sumar los elementos de su diagonal principal

    print("\n=== RESULTADO DE Marcos González ====")  # <- Poned aqui vuestros nombres!!
    # Código para mostrar por pantalla los resultados

    # Plot normalized confusion matrix
    tl.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                             title="Normalized confusion matrix")