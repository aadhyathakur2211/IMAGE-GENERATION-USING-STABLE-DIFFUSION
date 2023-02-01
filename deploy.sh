echo "Deploying"
vybe_gcloud.sh
gcloud config set project ai-ml-initiative
gcloud compute instances stop instance-1 --zone us-central1-a
gcloud compute instances start instance-1 --zone us-central1-a
