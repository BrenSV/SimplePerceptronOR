import java.util.Random;

/**
 * @author Brenda Sanchez Vazquez
 */

public class Perceptron {
    // Datos de entrada (x1, x2) y salidas esperadas (y)
    // Representan las combinaciones de entradas y las salidas correspondientes para la operación OR
    static int[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    static int[] outputs = {0, 1, 1, 1}; // Salida esperada para la operación lógica OR

    // Hiperparámetros del perceptrón
    static double learningRate = 0.1; // Tasa de aprendizaje, controla el tamaño de los ajustes de los pesos
    static double[] weights = new double[2]; // Pesos iniciales, uno por cada entrada
    static double bias; // Umbral inicial o peso asociado al sesgo
    static double convergenceThreshold = 0.001; // Cambio mínimo en los pesos para considerar convergencia
    static int maxEpochs = 1000; // Número máximo de épocas para evitar bucles infinitos

    /**
     * Función de activación escalón.
     * Convierte el valor de entrada calculado en una salida binaria.
     *
     * @param f Valor calculado de la función lineal.
     * @return 1 si f > 0.5, 0 en caso contrario.
     */
    public static int stepFunction(double f) {
        return f > 0.5 ? 1 : 0;
    }

    public static void main(String[] args) {
        System.out.println("Entrenamiento del Perceptrón con criterio de convergencia:");

        // Inicialización de pesos y bias con valores pequeños aleatorios
        Random random = new Random();
        for (int i = 0; i < weights.length; i++) {
            weights[i] = random.nextDouble() * 0.01; // Pesos pequeños iniciales
        }
        bias = random.nextDouble() * 0.01; // Umbral inicial aleatorio

        int epoch = 0; // Contador de épocas (iteraciones completas sobre los datos de entrenamiento)
        boolean converged = false; // Bandera para determinar si el entrenamiento ha convergido

        // Entrenamiento del perceptrón
        while (epoch < maxEpochs && !converged) {
            epoch++;
            System.out.println("Época " + epoch + ":");

            converged = true; // Asumimos inicialmente que no habrá más cambios significativos
            double maxWeightChange = 0.0; // Máximo cambio en los pesos en esta época

            // Iteración sobre cada ejemplo de entrenamiento
            for (int i = 0; i < inputs.length; i++) {
                // Calcular la salida del perceptrón (función lineal)
                // La función lineal combina las entradas ponderadas con sus pesos y suma un sesgo (bias).
                // Fórmula: f = (x1 * w1) + (x2 * w2) + bias
                // Donde:
                // - x1, x2: son las entradas actuales del ejemplo de entrenamiento.
                // - w1, w2: son los pesos asociados a las entradas x1 y x2, respectivamente.
                // - bias: es el sesgo que desplaza el resultado de la función para ajustar mejor la separación de clases.
                // El resultado "f" es el valor que se pasa a la función de activación para obtener l
                double f = inputs[i][0] * weights[0] + inputs[i][1] * weights[1] + bias;
                int yPred = stepFunction(f); // Aplicar la función de activación

                // Calcular el error: diferencia entre la salida esperada y la predicha
                int error = outputs[i] - yPred;

                // Si hay error, actualizar los pesos y el umbral
                if (error != 0) {
                    converged = false; // Si hay error, el modelo no ha convergido

                    // Actualización de los pesos
                    for (int j = 0; j < weights.length; j++) {
                        double weightChange = learningRate * error * inputs[i][j];
                        weights[j] += weightChange;
                        maxWeightChange = Math.max(maxWeightChange, Math.abs(weightChange));
                    }

                    // Actualización del umbral (bias)
                    bias += learningRate * error;
                }

                // Mostrar los detalles de la iteración
                System.out.println("  Entrada: [" + inputs[i][0] + ", " + inputs[i][1] +
                                   "], Salida Esperada: " + outputs[i] +
                                   ", Predicción: " + yPred + ", Error: " + error);
            }

            // Verificar convergencia basada en el cambio máximo de los pesos
            if (maxWeightChange > convergenceThreshold) {
                converged = false; // Si el cambio es mayor al umbral, no se ha alcanzado convergencia
            }

            // Mostrar los pesos y bias actuales después de la época
            System.out.println("  Pesos actuales: [" + weights[0] + ", " + weights[1] +
                               "], Umbral: " + bias);
        }

        // Mensaje de finalización del entrenamiento
        if (converged) {
            System.out.println("Convergencia alcanzada en " + epoch + " épocas.");
        } else {
            System.out.println("Se alcanzó el número máximo de épocas sin convergencia.");
        }

        // Validación del perceptrón: predicciones finales sobre los datos de entrada
        System.out.println("Validación del Perceptrón:");
        for (int i = 0; i < inputs.length; i++) {
            double f = inputs[i][0] * weights[0] + inputs[i][1] * weights[1] + bias;
            int yPred = stepFunction(f); // Predicción final
            System.out.println("Entrada: [" + inputs[i][0] + ", " + inputs[i][1] +
                               "], Predicción: " + yPred);
        }
    }
}