# Copy over ./id_rsa and ./id_rsa.pub to quaild-1.us-central1-a.angular-unison-350808
scp ./id_rsa ./id_rsa.pub user@quaild-1.us-central1-a.angular-unison-350808:/home/user/.ssh/

# Set permissions remotely
ssh user@quaild-1.us-central1-a.angular-unison-350808 "chmod 400 /home/user/.ssh/id_rsa"

# Clone the repository remotely
ssh user@quaild-1.us-central1-a.angular-unison-350808 "git clone git@github.com:Ghost---Shadow/quaild.git"

# Copy over ./.env file to quaild directory on remote server
scp ./.env user@quaild-1.us-central1-a.angular-unison-350808:/home/user/quaild/

# Copy over ./scratch file to quaild directory on remote server
scp ./scratch user@quaild-1.us-central1-a.angular-unison-350808:/home/user/quaild/

# Run source devops/install.sh remotely
ssh user@quaild-1.us-central1-a.angular-unison-350808 "cd quaild && source devops/install.sh"
