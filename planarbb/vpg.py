import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

class VPG:
    def __init__(self, input_size, output_size,
            gamma=0.99, alpha=0.003):

        # Hyperparams
        self.gamma = gamma
        self.alpha = alpha

        # Model
        x_in = tf.keras.Input([input_size,])
        x = tf.keras.layers.Dense(256, activation='relu')(x_in)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(output_size, activation='softmax')(x)

        self.model = tf.keras.Model(inputs=x_in, outputs=x, name='VPG')
        #self.model.summary()
        self.optimizer = tf.keras.optimizers.Adam(self.alpha)

        self.memory = [] # MRP: [(s0, a0, r1), (s1, a1, r2),...]

    def policy(self, state):
        probs = self.model.predict(np.array([state]))
        dist = tfp.distributions.Categorical(probs=probs, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])

    def store(self, state, action, reward):
        self.memory.append((state, action, reward))

    def learn(self):
            # Back-sample through episode for discounted returns
            G = 0
            returns = []
            for (_, _, reward) in reversed(self.memory):
                G = reward + self.gamma*G
                returns.append(G)
            returns.reverse()

            # Perform gradient ascent on policy at each time step
            with tf.GradientTape() as tape:
                loss = 0
                for (state, action, reward), G in zip(self.memory, returns):
                    probs = self.model(np.array([state]), training=True)

                    # Calculate loss
                    dist = tfp.distributions.Categorical(
                            probs=probs, dtype=tf.float32)
                    log_prob = dist.log_prob(action)
                    loss += -G*log_prob

            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                    zip(grads, self.model.trainable_variables))

            self.memory = []

