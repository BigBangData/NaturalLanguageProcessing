# Steps for Reproducibility

Here I'm assuming you have no Python and a Windows 10 machine. 

Mid-project I noticed my environments and package dependencies were such a mess I uninstalled EVERYTHING and started from scratch.

So here are the steps I took mid-project to crea a virtual environment for this project and get all the package dependencies into a requirements.txt file that can be imported for reproducibility.

---

## Steps

0. [OPTIONAL] Download and install [Git Bash](https://git-scm.com/downloads)
1. Download latest [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Check SHA256 hash on PowerShell, issue: `Get-FileHash <filename> -Algorithm SHA256` and compare hashes
3. Proceed to installation with default selections
4. Add Miniconda to the Path, go to: 
```
Control Panel > System and Security > System > Advanced System Settings > 
Environmental Variables > User Variables for <your_username> > Path > Edit > 
New > 
```
...and add your Miniconda path, should be similar to `C:\Users\<your_username>\miniconda3\`

5. In the project repo in Git Bash run: `conda init bash`
You should have (base) added to path, 

> Ex.
> ```
> (base)
> sanch@Cello MINGW64 ~/Documents/GitHub/NaturalLanguageProcessing (fuji)
> ```

6. Close, open again, then issue: `conda create -n py38` (or whatever name) to create your environment

7. Issue: `conda activate py38` to activate it, you should see (py38) added to path, 

> Ex.
> ```
> (py38)
> sanch@Cello MINGW64 ~/Documents/GitHub/NaturalLanguageProcessing (fuji)
> ```

- To deactivate issue: `conda deactivate`
- To remove environment: `conda remove -n <env_name> --all`

## Package Installations

8. Issue `conda install` in an activated env or `conda install -n <env_name>`, then:

> `jupyter notebook`
`numpy`
`pandas`
`matplotlib`
`nltk`
`scipy`
`scikit-learn`

- To uninstall packages: `conda remove scipy` OR `conda remove -n py38 scipy`
- To list packages: `conda list` 

9. To create a requirements.txt file for reproducibility: `conda list -e > requirements.txt`

- See the requirements.txt file for instructions on how to import it into a conda virtual environment.

10. To install packages not available in miniconda, use pip, I had to use it for: `pip install urlextract`

And that's it!

*Le Fin*

---









