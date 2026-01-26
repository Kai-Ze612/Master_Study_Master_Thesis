from setuptools import setup, find_packages

setup(
    name="ASAC",
    version="0.1.0",
    description="Augmented State Actor-Critic (A-SAC) for Robot Teleoperation",
    author="Kai",
    packages=find_packages(),  # This automatically finds the 'ASAC' folder
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "torch",
        "gymnasium",
        "mujoco",
        "stable-baselines3",
        "tensorboard",
        "tqdm",
    ],
)