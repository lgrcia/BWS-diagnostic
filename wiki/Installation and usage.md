## Installation - For Python begineers 

_For **Python-fluent** people: [skip](Package installation)_

### Python distribution: Anaconda <img src="images/anaconda_logo.png" width="140" align=right> 

The first needed is a proper **Pyhton distribution**. **[Anaconda](https://www.continuum.io/downloads)** is a freemium open source distribution of the Python and R programming languages for large-scale data processing, predictive analytics, and scientific computing, that aims to simplify package management and deployment. [Wikipedia](https://en.wikipedia.org/wiki/Anaconda_(Python_distribution))

### IDE: Pycharm <img src="images/pycharm_logo.png" width="100" align=right>

Python files are just text files. But to read and write Pyhton code in a more comfortable way, an **IDE** (**integrated development environment**) is required. **Spyder** is a good one to start, especially for people used to Matlab, but **[PyCharm](https://www.jetbrains.com/pycharm/download)** provide a more professional environment, easing the managment of large projects. PyCharm also provide a set of intersting integrated tools such as SVN, Git versionning or easy virtual environment management. Forget about it if you don't know it but it could be usefull in your later works.

After installing Pycharm, you will need to follow this procedure to specify that you want to use the Python distribution provided by Anaconda (if you have multiple verison of Pyhton on your computer).

### Package installation

**BWS-diagnostic** relies on the packages available in the Anaconda distribution. Here is a non axhaustive list of the packages use by the library:

- Matplotlib
- Scipy
- Sklearn
- Numpy

And here is some package missing that need to be installed separately:

- [npTdms](https://pypi.python.org/pypi/npTDMS): for tdms file management
- [tqdm](https://pypi.python.org/pypi/tqdm/4.15.0): for console loading bars
- [configparser](https://pypi.python.org/pypi/configparser/3.5.0): for configuration file managment




### For Python begineers


To install a new package you can follow the procedure of the PyCharm IDE or isntall it manually using :

```
pip install package_name
```

Then you just have to import the package to use it in your code:

```python
import package_name
```

To use BWS-diagnotic just open the complete folder in PyCharm and go to `script/test_bench`, here you can import all the tools provided by the library. For example

```Python
from lib import ops_processing as ops
from lib import diaqnostic_tool as dt
from lib import prairie 

#Your code here
```

and you are done :+1:









