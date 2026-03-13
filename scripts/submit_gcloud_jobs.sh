#!/bin/bash
# Script to submit multiple training jobs to GCloud Batch

set -e

PROJECT_ID="project-42e015a5-9baf-49f2-8da"
IMAGE_URI="gcr.io/$PROJECT_ID/reg-transfo:latest"
REGION="us-central1"  # Adapter selon ta région préférée

# Liste des expériences
EXPERIMENTS=(
    "gnnvitqm9"
    "gnnvitqm8"
    "gnnvitqm7"
    # Ajoute d'autres expériences ici
)

echo "🚀 Submitting GCloud Batch jobs..."
echo "Project: $PROJECT_ID"
echo "Image: $IMAGE_URI"
echo ""

for exp in "${EXPERIMENTS[@]}"; do
    job_name="reg-transfo-${exp,,}"  # Lowercase

    echo "📝 Submitting job: $job_name (experiment=$exp)"

    gcloud batch jobs submit "$job_name" \
        --location="$REGION" \
        --config - <<EOF
{
  "taskGroups": [
    {
      "taskSpec": {
        "runnables": [
          {
            "container": {
              "imageUri": "$IMAGE_URI",
              "commands": [
                "/bin/bash",
                "-c"
              ],
              "args": [
                ". .venv/bin/activate && python -m reg_transfo.main experiment=$exp"
              ]
            }
          }
        ],
        "computeResource": {
          "cpuMilli": 4000,
          "memoryMib": 16384
        },
        "maxRetries": 2,
        "timeout": "86400s"
      },
      "taskCount": 1,
      "parallelism": 1
    }
  ],
  "logsPolicy": {
    "destination": "CLOUD_LOGGING"
  }
}
EOF

    echo "✓ Job $job_name submitted!"
    echo ""
done

echo "✅ All jobs submitted! Check status with:"
echo "   gcloud batch jobs list --location=$REGION"
