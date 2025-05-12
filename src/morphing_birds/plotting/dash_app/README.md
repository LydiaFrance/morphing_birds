## Dash App to view combinations of flight principal components.

The Dash app included in this directory (`app.py`) provides an interface with a drop-down menu where users can select combinations of principal components. Upon pressing play, the interactive plot will show a model of a hawk flying demonstrating these particular components.

In order for this app to function correctly, ensure the following files are in the `data` directory in the root of the repository. These are:
1. `mean_hawk_shape.csv`
2. `website_principal_components.npy`
3. `Left_scores_RightTurn.npy`
4. `Right_scores_RightTurn.npy`
5. `website_mu.npy`

### To run the app locally.

To run the app on a local webserver in debug mode:

`python src/morphing_birds/plotting/dash_app/app.py`

The app will then be available to view on a localhost port, normally http://127.0.0.1:8050/ .

### To deploy the app the Google App Engine (GAE).

These are the steps used to deploy the application to the cloud using GAE.

1. Activate a Google Cloud account.
2. Created a project e.g. `hawk-wing-movement`.
3. Install the gcloud command-line-interface (CLI) according to the instructions [here](https://cloud.google.com/sdk/docs/install).
4. Log into the project using `gcloud init` and your Google credentials.
4. Set up GAE using `gcloud app create` and choose the `europe-west-2` region.
5. The `app.yaml` file in the root of the `morphing-birds` project contains configuration details for the app.
6. There is also a `requirements.txt` file in the root of the repository which is used during deployment.
7. To prevent permissions issues when deploying the app run the following commands from the CLI:
```shell
gcloud projects add-iam-policy-binding hawk-wing-movement \
    --member "serviceAccount:hawk-wing-movement@appspot.gserviceaccount.com" \
    --role "roles/storage.admin"
```

```shell
gcloud projects add-iam-policy-binding hawk-wing-movement \
    --member "serviceAccount:hawk-wing-movement@appspot.gserviceaccount.com" \
    --role "roles/artifactregistry.reader"
```

```shell
gcloud projects add-iam-policy-binding hawk-wing-movement \
    --member "serviceAccount:hawk-wing-movement@appspot.gserviceaccount.com" \
    --role "roles/artifactregistry.writer"
``` 

```shell
gcloud projects add-iam-policy-binding hawk-wing-movement \
    --member "serviceAccount:hawk-wing-movement@appspot.gserviceaccount.com" \
    --role "roles/artifactregistry.createOnPushWriter"
```
8. Finally, to deploy the app, `cd` to the root of the repository (containing the `app.yaml` file) and run `gcloud app deploy`.
