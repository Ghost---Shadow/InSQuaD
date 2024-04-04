export ZONE=$(gcloud compute instances list $HOSTNAME --format 'csv[no-heading](zone)')

echo "Stopping $HOSTNAME in $ZONE"
gcloud compute instances stop $HOSTNAME --zone=$ZONE
