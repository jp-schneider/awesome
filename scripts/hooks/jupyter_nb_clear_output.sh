#!/usr/bin/bash
#!/usr/bin/env bash
# Cleaning cell outputs of jupyter notebooks to avoid large disk usage in git repos.
PROJECT_DIR=$PWD

# Load profile if exists to enable poetry
if [ -e $HOME/.profile ]
then
    . $HOME/.profile
fi
cd $PROJECT_DIR

for FILE in "$@"
do
    echo "Cleaning cells on: $FILE"

    # Cleaning them with nbconvert

    poetry run python -m jupyter nbconvert --clear-output --inplace "$PROJECT_DIR/$FILE"

    # Done
    echo "Cleaned $FILE!"
done

exit 0
