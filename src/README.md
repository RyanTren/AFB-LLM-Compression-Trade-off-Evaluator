# LoRA LLM Compression Technique Branch

## Overview
This project is a Python application packaged with Docker. It includes a set of scripts that implement the core functionality of the application, along with Docker configuration files to facilitate containerization and deployment.

## Project Structure
```
src
├── docker
│   ├── Dockerfile
│   └── docker-compose.yml
├── scripts
│   ├── __init__.py
│   ├── main.py
│   └── helper.py
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

## Installation
To set up the project, ensure you have Docker and Docker Compose installed on your machine. Clone the repository and navigate to the project directory.

```bash
git clone <repository-url>
cd docker-python-project
```

## Usage
To build and run the application, use the following command:

```bash
docker-compose up --build
```

This command will build the Docker image as specified in the `Dockerfile` and start the application as defined in the `docker-compose.yml` file.

## Scripts
- **main.py**: The main entry point of the application.
- **helper.py**: Contains utility functions used by the main application.

## Dependencies
The project dependencies are listed in the `requirements.txt` file. Make sure to install them before running the application.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.