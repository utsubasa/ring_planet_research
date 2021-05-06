## Learning lightcurve analysis (work at ./ipynb)

## 1. Prepartion
-  Install python & Jupyter  
-  Anaconda may be the easiest way to install python, Jupyter, and other modules  
https://docs.anaconda.com/anaconda/install/


## 2. Install python modules
### 2.1 List of requisite modules
#### 2.1.1 lightkurve @v2.0.7 (https://docs.lightkurve.org/quickstart.html)  
"conda install" may give you old version (v1), so you had better install it using pip.  
It will also automatically install astropy, which is used in Tutorial 4 just once. 
#### 2.1.2 batman (https://lweb.cfa.harvard.edu/~lkreidberg/batman/)  
if you have problem w/ gcc, you may had better downgrade Xcode to 11.5 from 12.X. 
#### 2.1.3 lmfit (https://lmfit.github.io/lmfit-py/)  
#### 2.1.4 numpy, matplotlib, scipy **(automatically installed through anaconda)**

### 2.2 how to install modules
#### (a) If you want to install ##,  
conda install ## 

#### (b) If modules cannot be install by (a), search in another repository,  
conda install -c conda-forge ## (to see another repository.)  
 
#### (c) If modules cannot be install by (a) & (b), use pip
pip install ##  
(pip & anaconda are sometimes interfered with each other, so you should be careful about that. )

## 3. Tutorials
### 3.1 Test for fitting with LM method 
run "Tutorial1_test_for_LM_fitting.ipynb"
### 3.2 Test for transit calculation (Tutorial2_test_for_transit_calc.ipynb)
run Tutorial2_test_for_transit_calc.ipynb
### 3.3 Test for fitting transit light curves with LM method  
run "Tutorial3_transit_fitting.ipynb"
### 3.4 Analyze transits for Pi Mensae (first TESS planet)
run "Tutorial4_pi_mensae.ipynb"
