dcptree
========



## Installation
  
Run the following snippet to install ``dcptree`` in a Mac/Unix environment, and complete a test run.  

```
git clone https://github.com/ustunb/dcptree
cd dcptree
pip install -e . #install in editable mode  
```

## Requirements


### CPLEX

CPLEX is cross-platform commercial optimization tool with a Python API. It is freely available to students and faculty members at accredited institutions as part of the IBM Academic Initiative. 

To get CPLEX:

1. Sign up for the [IBM Academic Initiative](https://developer.ibm.com/academic/). Note that it may take up to a week to obtain approval.
2. Download *IBM ILOG CPLEX Optimization Studio* from the [software catalog](https://ibm.onthehub.com/WebStore/OfferingDetails.aspx?o=6fcc1096-7169-e611-9420-b8ca3a5db7a1)
3. Install the file on your computer. Note mac/unix users will [need to install a .bin file](http://www-01.ibm.com/support/docview.wss?uid=swg21444285).
4. Setup the CPLEX Python API [as described here here](http://www.ibm.com/support/knowledgecenter/SSSA5P_12.6.3/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html).

If you have problems installing CPLEX, check the [CPLEX user manual](http://www-01.ibm.com/support/knowledgecenter/SSSA5P/welcome) or the [CPLEX forums](https://www.ibm.com/developerworks/community/forums/html/forum?id=11111111-0000-0000-0000-000000002059). 
