import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.patches import Circle
import os

#TDDL Fix the diffusion term
#note to me D = 800 where the full eq is 2 *D = omage^2 / mu , mu =2500.0, omaga = 2000.0

def getbeta(alpha):
    return(2*alpha/(1+alpha))
# Define the true solution
def true_solution(x, y):
    radius = np.sqrt(x**2 + y**2)
    bigradius = 5 
    smradius = 1 
    return -1/(4*D)*radius**2 + 1/(4*D)*(bigradius**beta*smradius**2-bigradius**2 *smradius**beta)/(bigradius**beta - smradius**beta)+(radius**beta)/(4*D)*(bigradius**2-smradius**2)/(bigradius**beta - smradius**beta)

# PINN model definition
class PINN(tf.keras.Model):
    def __init__(self, hidden_units=64, hidden_layers=5):
        super(PINN, self).__init__()
        
        self.hidden_layers = [
            tf.keras.layers.Dense(hidden_units, activation='swish', kernel_initializer='glorot_normal') for _ in range(hidden_layers)
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
        r = tf.sqrt(x**2 + y**2)

        # Boundary condition factors (vanish at r=1 and r=5)
        bc_factor = (r - 1) * (5 - r)

        # Enforce boundary conditions in the output (zero on the circle boundary)
        return bc_factor * f 

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
def loss_fn(model, x_train, y_train, x_boundary, y_boundary, T_boundary,train_data):
    # PDE residual loss
    residual = pde_residual(model, x_train, y_train)
    pde_loss = tf.reduce_mean(tf.math.log(1 + tf.square(residual )))


    # Boundary condition loss
    T_pred_boundary = model(tf.stack([x_boundary, y_boundary], axis=1))
    boundary_loss = tf.reduce_mean(tf.square(T_pred_boundary - T_boundary))

    scale_factor = tf.reduce_mean(tf.abs(residual)) / (tf.reduce_mean(tf.abs(T_boundary)) + 1e-8)
    total_loss = pde_loss + boundary_loss*scale_factor
    return total_loss, pde_loss, boundary_loss

# Training function
def train_pinn(model, optimizer, epochs, x_train, y_train, x_boundary, y_boundary, T_boundary,train_data):
    loss_history = {"total_loss": []}
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            total_loss, pde_loss, boundary_loss = loss_fn(
                model, x_train, y_train, x_boundary, y_boundary, T_boundary,train_data
            )
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        loss_history["total_loss"].append(total_loss.numpy())
        

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, PDE Loss: {pde_loss.numpy()}, Boundary Loss: {boundary_loss.numpy()}, Total Loss: {total_loss.numpy()}")
    return loss_history   

def sample_circle_domain(n_points):
    inner_radius=1
    outer_radius=5
    SEED = 10
    mid_radius = (inner_radius+outer_radius)/2
    # Sample radius in the annular region
    r = np.random.normal(loc=mid_radius, scale=(outer_radius - inner_radius) / 4, size=n_points)
    
    # Clip to ensure radius stays within the valid range
    r = np.clip(r, inner_radius, outer_radius)
    
    # Sample angles uniformly
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    
    # Convert polar coordinates to Cartesian
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    return tf.convert_to_tensor(x, dtype=tf.float32), tf.convert_to_tensor(y, dtype=tf.float32)


def sample_circle_boundary(n_points, radii=[1, 5]):
    x_list = []
    y_list = []

    for r in radii:
        theta = np.random.uniform(0, 2 * np.pi, n_points)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        x_list.append(x)
        y_list.append(y)

    # Concatenate points from all boundaries
    x_combined = np.concatenate(x_list)
    y_combined = np.concatenate(y_list)

    return tf.convert_to_tensor(x_combined, dtype=tf.float32), tf.convert_to_tensor(y_combined, dtype=tf.float32)

def training_data(x, y, noise_std=0.00008):
    radius = np.sqrt(x**2 + y**2)
    SEED = 2
    
    tf.random.set_seed(SEED)
    # Calculate ans
    ans = (-1 / (4 * D) * radius**2 + 
           1 / (4 * D) * (bigradius**beta * smradius**2 - bigradius**2 * smradius**beta) / (bigradius**beta - smradius**beta) + 
           (radius**beta) / (4 * D) * (bigradius**2 - smradius**2) / (bigradius**beta - smradius**beta))
    
    # Add noise
    noise = tf.random.normal(ans.shape, mean=0.0, stddev=noise_std)
    ans_with_noise = ans 
    
    # Clip to ensure positive values
    ans_positive = tf.maximum(ans_with_noise, 0.0)
    return ans_positive



# Define constants



bigradius = 5
smradius = 1 
alpha = .001 
D = 400  # Diffusion coefficient
beta = getbeta(alpha)

x_train, y_train = sample_circle_domain(1000)  # Points inside the circle

train_data = training_data(x_train ,y_train)

x_boundary, y_boundary = sample_circle_boundary(100)  # Points on the boundary
T_boundary = tf.zeros_like(x_boundary)  # Boundary condition: T = 0

# Initialize and train the model
model = PINN()
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0033, #so far .0033 gave the best results, I have tried [.032,.033,.0034,.035]
    beta_1=0.9,#default is .9
    beta_2=0.999,#default is .999
    epsilon=1e-7#default is 1e-7
)

loss_history = train_pinn(
    model, optimizer, epochs=5000, 
    x_train=x_train, y_train=y_train, 
    x_boundary=x_boundary, y_boundary=y_boundary, 
    T_boundary=T_boundary,
    train_data= train_data
)


# Generate a grid of points inside the circle
x_grid = np.linspace(-5, 5, 500)
y_grid = np.linspace(-5, 5, 500)
x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

mask = x_mesh**2 + y_mesh**2 <= 5**2  # Points inside the larger circle
mask2 = x_mesh**2 + y_mesh**2 >= 1**2  # Points outside the smaller circle
x_circle = x_mesh[mask & mask2]  # Keep points between the two circles
y_circle = y_mesh[mask & mask2]

# Stack the x and y coordinates and predict using the model
xy_points = np.stack([x_circle, y_circle], axis=1)
T_pred_circle = model(tf.convert_to_tensor(xy_points, dtype=tf.float32)).numpy()


T_true_circle = true_solution(x_circle,y_circle).flatten()


# Create a full grid of predictions with NaNs outside the circle for contour plotting
T_full = np.full_like(x_mesh, np.nan, dtype=np.float32)
T_full[mask & mask2] = T_pred_circle.flatten()

L2_error = np.sqrt(np.mean((T_full[mask & mask2] - T_true_circle) ** 2))
diff = np.abs(T_full[mask & mask2] - T_true_circle)
print("L2 Error:", L2_error)
path = "/Users/jacobmantooth/Desktop/mtstuff/figs/PINNBOTHDC"
if (L2_error< .001):
    plt.figure(figsize=(10, 6))
    plt.scatter(x_circle, y_circle, c=diff, cmap='viridis')
    plt.colorbar(label='Absolute Error')
    plt.title('Pointwise Absolute Error')
    file_name = f"Pointwise_Absolute_Error_{L2_error:.4f}.png" 
    plt.savefig(os.path.join(path, file_name))
    plt.show()
    


    # Plot the contour of predicted temperature over the circle
    fig, ax = plt.subplots(figsize=(10, 6))  # Set the figure size here

    # Contour plot
    contour = ax.contourf(x_mesh, y_mesh, T_full, cmap="viridis")
    plt.colorbar(contour, ax=ax, label="Predicted T(x, y)")
    
    # Add circles
    circle1 = Circle((0, 0), radius=1, fill=False, edgecolor='red', linewidth=2)  # Circle with radius 1
    circle2 = Circle((0, 0), radius=5, fill=False, edgecolor='red', linewidth=2)   # Circle with radius 5
    ax.add_patch(circle1)
    ax.add_patch(circle2)

    # Ensure aspect ratio is equal and set limits
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)

    # Titles and labels
    ax.set_title("Contour Plot of Predicted T(x, y) Over the Circle")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    file_name = f"Predicted_T_{L2_error:.4f}.png" 
    plt.savefig(os.path.join(path, file_name))
    plt.show()


    plt.figure(figsize=(10, 6))
    plt.plot(loss_history["total_loss"], label="Total Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Losses Over Epochs")
    plt.legend()
    file_name = f"Total_Loss_{L2_error:.4f}.png" 
    plt.savefig(os.path.join(path, file_name))
    plt.show()
    
    # Predict and plot results
    x_plot = np.linspace(1.01, 4.99, 100)
    y_plot = np.zeros(np.size(x_plot))
    T_pred = model(tf.convert_to_tensor(np.stack([x_plot, y_plot], axis=1), dtype=tf.float32)).numpy()
    T_exact = true_solution(x_plot, y_plot)


    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, T_exact, label="Exact Solution")
    plt.plot(x_plot, T_pred.flatten(), label="PINN Prediction", linestyle="dashed")
    plt.xlabel("x")
    plt.ylabel("T(x, y=0)")
    plt.legend()
    plt.title("Comparison of PINN Prediction and Exact Solution")
    file_name = f"Prediction_Exact_LeftLineSolution_{L2_error:.4f}.png" 
    plt.savefig(os.path.join(path, file_name))
    plt.show()
    


    x_plot = np.linspace(-4.99,-1.01, 100)
    y_plot = np.zeros(np.size(x_plot))
    T_pred = model(tf.convert_to_tensor(np.stack([x_plot, y_plot], axis=1), dtype=tf.float32)).numpy()
    T_exact = true_solution(x_plot, y_plot)


    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, T_exact, label="Exact Solution")
    plt.plot(x_plot, T_pred.flatten(), label="PINN Prediction", linestyle="dashed")
    plt.xlabel("x")
    plt.ylabel("T(x, y=0)")
    plt.legend()
    plt.title("Comparison of PINN Prediction and Exact Solution")
    file_name = f"Prediction_Exact_RightLineSolution_{L2_error:.4f}.png" 
    plt.savefig(os.path.join(path, file_name))
    plt.show()
    
   


    optimizer_config = {
    "learning_rate": optimizer.learning_rate.numpy(),  # Convert to a value for saving
    "beta_1": optimizer.beta_1,
    "beta_2": optimizer.beta_2,
    "epsilon": optimizer.epsilon
    }


    with open(f"{path}/Optimizer_Config{L2_error}", 'a') as file2write:
        file2write.write("Optimizer configuration:\n")
        for key, value in optimizer_config.items():
            file2write.write(f"{key}: {value}\n")
