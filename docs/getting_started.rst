Getting Started with GoFast Development
=======================================

Welcome to the GoFast development community! This guide will help you set up your development environment and get started with contributing to GoFast.

Setting Up Your Development Environment
---------------------------------------

1. **Install Python**: Ensure you have Python 3.6 or later installed on your system. You can download Python from `https://www.python.org/downloads/`.

2. **Fork the GoFast Repository**: Go to `https://github.com/WEgeophysics/gofast` and fork the repository to your GitHub account.

3. **Clone Your Fork**: Clone the forked repository to your local machine using:

   .. code-block:: bash

       git clone https://github.com/yourusername/gofast.git
       cd gofast

4. **Create a Virtual Environment** (optional, but recommended):

   .. code-block:: bash

       python -m venv gofast_env
       source gofast_env/bin/activate  # On Windows use `gofast_env\Scripts\activate`

5. **Install Dependencies**:

   .. code-block:: bash

       pip install -r requirements.txt

Understanding the Git Workflow
------------------------------

- **Branching**: Create a new branch for each feature or fix. Name your branches meaningfully.

  .. code-block:: bash

      git checkout -b your-branch-name

- **Making Changes**: Implement your changes, adhering to the code style and quality guidelines of GoFast.

- **Committing Changes**: Commit your changes with clear, concise messages.

  .. code-block:: bash

      git add .
      git commit -m "Your commit message"

- **Pushing Changes**: Push your changes to your forked repository.

  .. code-block:: bash

      git push origin your-branch-name

- **Creating a Pull Request**: Go to the GoFast GitHub repository and open a pull request for your branch.

Please read the `CONTRIBUTING.rst` file for more detailed instructions on contributing to GoFast.
