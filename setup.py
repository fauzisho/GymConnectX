from setuptools import setup, find_packages

setup(
    name='gymconnectx',
    version='1.0.5',
    description='ConnectX is a game for two players that is based on the well-known Connect 4. The goal is to place X coins in a row, column, or diagonal on a board with dimensions M by N.',
    url='https://github.com/fauzisho/GymConnectX',
    author='Fauzi Sholichin',
    license='MIT License',
    packages=find_packages(),
    install_requires=['gym', 'pygame', 'numpy'],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
    keywords=["tictactoe", "gym", "pygame"],
    python_requires=">=3.9",
    include_package_data=True,
)