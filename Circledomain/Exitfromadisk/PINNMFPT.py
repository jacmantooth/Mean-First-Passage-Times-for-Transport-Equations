import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 

# Define constants
mu = 2000000  # Mobility
v = 40000  # Speed
outrad = 1  # Outer radius
D = 800  # Diffusion coefficient

# Define the true solution
def true_solution(x, y):
    radius = np.sqrt(x**2 + y**2)
    return 1 / (4 * D) * (outrad**2 - radius**2)

# Define the coefficients
def D11(x, y):
    return 0.5 *D

def D12(x, y):
    return 0.0

def D22(x, y):
    return 0.5 *D

# PINN model definition
class PINN(tf.keras.Model):
    def __init__(self, hidden_units=64, hidden_layers=5):
        super(PINN, self).__init__()
        self.hidden_layers = [
            tf.keras.layers.Dense(hidden_units, activation='tanh', kernel_initializer='glorot_normal') for _ in range(hidden_layers)
        ]
        self.output_layer = tf.keras.layers.Dense(1, activation=None)
    def call(self, inputs):
        # Split inputs into x and y components
        x, y = inputs[:, 0], inputs[:, 1]
        
        # Forward pass through the hidden layers
        z = inputs
        for layer in self.hidden_layers:
            z = layer(z)
        
        f = self.output_layer(z)  # Shape: (batch_size, 1)

        # Ensure x and y have the correct shape for broadcasting
        x = tf.expand_dims(x, axis=-1)  # Shape: (batch_size, 1)
        y = tf.expand_dims(y, axis=-1)  # Shape: (batch_size, 1)

        # Enforce boundary conditions in the output (zero on the circle boundary)
        return (1 - x**2 - y**2) * f



def pde_residual(model, x, y):
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([x, y])
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([x, y])
            T = model(tf.stack([x, y], axis=1))  # Predict T(x, y)

        # First derivatives
        dT_dx = tape1.gradient(T, x)
        dT_dy = tape1.gradient(T, y)

    # Second derivatives
    d2T_dx2 = tape2.gradient(dT_dx, x)
    d2T_dy2 = tape2.gradient(dT_dy, y)

    # PDE residual
    residual = D * (d2T_dx2 + d2T_dy2) + 1
    return residual

# Loss function for PDE and boundary conditions
def loss_fn(model, x_train, y_train, x_boundary, y_boundary, T_boundary):
    # PDE residual loss
    residual = pde_residual(model, x_train, y_train)
    pde_loss = tf.reduce_mean(tf.square(residual))

    # Boundary condition loss
    T_pred_boundary = model(tf.stack([x_boundary, y_boundary], axis=1))
    boundary_loss = tf.reduce_mean(tf.square(T_pred_boundary - T_boundary))

    total_loss = pde_loss + 100 * boundary_loss
    return total_loss, pde_loss, boundary_loss

# Training function
def train_pinn(model, optimizer, epochs, x_train, y_train, x_boundary, y_boundary, T_boundary):
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            total_loss, pde_loss, boundary_loss = loss_fn(
                model, x_train, y_train, x_boundary, y_boundary, T_boundary
            )
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, PDE Loss: {pde_loss.numpy()}, Boundary Loss: {boundary_loss.numpy()}, Total Loss: {total_loss.numpy()}")
def sample_circle_domain(n_points):
    """Generate random points inside the unit circle."""
    r = np.sqrt(np.random.uniform(0, 1, n_points))  # Radius
    theta = np.random.uniform(0, 2 * np.pi, n_points)  # Angle
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return tf.convert_to_tensor(x, dtype=tf.float32), tf.convert_to_tensor(y, dtype=tf.float32)

def sample_circle_boundary(n_points):
    """Generate random points on the unit circle boundary."""
    theta = np.linspace(0, 2 * np.pi, n_points)
    x = np.cos(theta)
    y = np.sin(theta)
    return tf.convert_to_tensor(x, dtype=tf.float32), tf.convert_to_tensor(y, dtype=tf.float32)




x_train, y_train = sample_circle_domain(1000)  # Points inside the circle
x_boundary, y_boundary = sample_circle_boundary(100)  # Points on the boundary
T_boundary = tf.zeros_like(x_boundary)  # Boundary condition: T = 0

# Initialize and train the model
model = PINN()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

train_pinn(
    model, optimizer, epochs=10000, 
    x_train=x_train, y_train=y_train, 
    x_boundary=x_boundary, y_boundary=y_boundary, 
    T_boundary=T_boundary
)

# Predict and plot results
x_plot = np.linspace(-0.99, 0.99, 100)
y_plot = np.zeros_like(x_plot)
T_pred = model(tf.convert_to_tensor(np.stack([x_plot, y_plot], axis=1), dtype=tf.float32)).numpy()
T_exact = true_solution(x_plot, y_plot)

plt.figure(figsize=(10, 6))
plt.plot(x_plot, T_exact, label="Exact Solution")
plt.plot(x_plot, T_pred, label="PINN Prediction", linestyle="dashed")
plt.xlabel("x")
plt.ylabel("T(x, y=0)")
plt.legend()
plt.title("Comparison of PINN Prediction and Exact Solution")
plt.show()

# Generate a grid of points inside the circle
x_grid = np.linspace(-1, 1, 200)
y_grid = np.linspace(-1, 1, 200)
x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

# Filter points to keep only those inside the circle
mask = x_mesh**2 + y_mesh**2 <= 1
x_circle = x_mesh[mask]
y_circle = y_mesh[mask]

# Stack the x and y coordinates and predict using the model
xy_points = np.stack([x_circle, y_circle], axis=1)
T_pred_circle = model(tf.convert_to_tensor(xy_points, dtype=tf.float32)).numpy()

# Create a full grid of predictions with NaNs outside the circle for contour plotting
T_full = np.full_like(x_mesh, np.nan, dtype=np.float32)
T_full[mask] = T_pred_circle.flatten()

# Plot the contour of predicted temperature over the circle
plt.figure(figsize=(8, 8))
contour = plt.contourf(x_mesh, y_mesh, T_full, levels=50, cmap="viridis")
plt.colorbar(contour, label="Predicted T(x, y)")
plt.title("Contour Plot of Predicted T(x, y) Over the Circle")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()
