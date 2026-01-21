# %%
# %%
# Criar dataset simples
# Exemplo para o Leandro: Criar um classificador OR, um tipo básico de operação de computador
# O OR retorna 1 qaundo qualquer item seja 1 também
import numpy as np

ops = {
    (1, 0): 1,
    (0, 1): 1,
    (1, 1): 1,
    (0, 0): 0
}

training_inputs = np.array(list(ops.keys()))
training_targets = np.array(list(ops.values()))

# A variável ops cria os cenário possíveis

# %%
# Cria um neuronio artificial simples
import numpy as np

# classe base do nosso neurônio, basicamente um neuronio tem pesos (w), bias (b) e o rate learning (p)
# peso -> O valor que vai ser modificado nos treinamentos
# bias -> A variância do neuronio, tende a modificar a resposta
# rate learning -> a taxa de aprendizado
class Neuron:
    def __init__(self, n_inputs: int, rate_learning: float):
        self.w = np.random.rand(n_inputs)
        self.b = np.random.rand()
        self.p = rate_learning
    
    def activation_function(self, x):
        """Aqui é a função de ativação, ela que é responsável por gerar o valor final do neurônio e de fato entender padroes"""
        # Leandro, aqui vou usar a função "degrau" que basicamente só ativa se o valor for 1, perfeito para o exemplo do xor
        # veja também RELU, sigmoid e tanh, outras funções de ativação

        if x >= 0:
            return 1
        return 0
    
    def predict(self, inputs: list[int]):
        """O método que preve o resultado baseado nos inputs"""
        psum = np.dot(inputs, self.w) + self.b
        return self.activation_function(psum)

    def train(self, training_inputs, training_targets, epochs):
        """Ajusta pesos e bias com base no erro (Regra de Hebb/Perceptron)."""
        # Leandro, não se perca muito nesse código, mesmo sendo simples, normalmente frameworks usam algoritmos mais complexos, só entenda como funciona um treinamento simples
        for _ in range(epochs):
            for inputs, target in zip(training_inputs, training_targets):
                prediction = self.predict(inputs)
                error = target - prediction
                
                # Ajuste dos parâmetros: novo_valor = valor + (error * entrada * taxa)
                self.w += error * inputs * self.p
                self.b += error * self.p

# %%
# Exemplo de uso

# usarei uma taxa de 0.1 de aprendizado, enquanto menor a taxa, mais fino será o ajuste de erro.
simple_neuron = Neuron(2, 0.1)

# vamos treinar usando 100 epocas
simple_neuron.train(training_inputs, training_targets, epochs=100)

# após treinado, vamos testar essa bagaça
simple_neuron.predict([0, 0])