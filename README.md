This repository contains a software implementation of the methods described in:

[**Fairness without Harm: Decoupled Classifiers with Preference Guarantees**](http://proceedings.mlr.press/v97/ustun19a.html)    
Berk Ustun, Yang Liu, David Parkes   
International Conference on Machine Learning (ICML), 2019

---

## Installation

#### Requirements

- Python 3
- CPLEX


#### CPLEX

CPLEX is cross-platform commercial optimization tool with a Python API. It is free for students and faculty at accredited institutions. To get CPLEX:

1. Register for [IBM OnTheHub](https://ibm.onthehub.com/WebStore/Account/VerifyEmailDomain.aspx)
2. Download the *IBM ILOG CPLEX Optimization Studio* from the [software catalog](https://ibm.onthehub.com/WebStore/ProductSearchOfferingList.aspx?srch=CPLEX)
3. Install the CPLEX Optimization Studio.
4. Setup the CPLEX Python API [as described here](https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html).

If you have problems installing CPLEX, check the [CPLEX user manual](http://www-01.ibm.com/support/knowledgecenter/SSSA5P/welcome) or the [CPLEX forums](https://www.ibm.com/developerworks/community/forums/html/forum?id=11111111-0000-0000-0000-000000002059). 
 
  
## Development

The code in this repository is currently under active development, and may therefore change substantially with each commit.   
   
## Reference

If you end up using our code, please cite the following paper: 


```
@InProceedings{pmlr-v97-ustun19a,
  title = 	 {Fairness without Harm: Decoupled Classifiers with Preference Guarantees},
  author = 	 {Ustun, Berk and Liu, Yang and Parkes, David},
  booktitle = 	 {Proceedings of the 36th International Conference on Machine Learning},
  pages = 	 {6373--6382},
  year = 	 {2019},
  editor = 	 {Chaudhuri, Kamalika and Salakhutdinov, Ruslan},
  volume = 	 {97},
  series = 	 {Proceedings of Machine Learning Research},
  address = 	 {Long Beach, California, USA},
  month = 	 {09--15 Jun},
  publisher = 	 {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v97/ustun19a/ustun19a.pdf},
  url = 	 {http://proceedings.mlr.press/v97/ustun19a.html},
}
```
