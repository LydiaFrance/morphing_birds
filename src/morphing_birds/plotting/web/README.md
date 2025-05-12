## Static Webpage Plots of Bird Flight

The plot allows selection of Principal Components for bird flight movement and is displayed on a static website page, suitable for hosting on Github pages.

To run the script to generate a new `index.html` file:
Firstly ensure you have added the required data files to the `data` directory in the root of the repository. These are:
1. `mean_hawk_shape.csv`
2. `website_principal_components.npy`
3. `website_score_10frames.npy`
4. `website_mu.npy`

Then run the script:
`python src/morphing_birds/plotting/web/output_template_plots.py`

This will create an HTML file in the `src/morphing_birds/plotting/web` directory based upon the template found at `src/morphing_birds/plotting/web/template.html` 

When happy with the `index.html` file, it can be copied into the root directory of the repository for serving on GitHub pages.
