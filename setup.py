from setuptools import setup

setup(name='RewardLearning',
      version='0.1.0',
      description="Reward Learning project",
      author="Abdalkarim Mohtasib",
      author_email='amohtasib@lincoln.ac.uk',
      platforms=["any"],
      license="GPLv3",
      url="https://github.com/Mohtasib/RewardLearning",
      install_requires=["Keras>=2.0.0"],
      extras_require={
            # These requires provide different backends available with Keras
            "tensorflow": ["tensorflow"]
      }
)
