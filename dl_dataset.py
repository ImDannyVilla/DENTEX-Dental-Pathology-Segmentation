import kagglehub

# Download latest version
path = kagglehub.dataset_download("truthisneverlinear/dentex-challenge-2023")

print("Path to dataset files:", path)