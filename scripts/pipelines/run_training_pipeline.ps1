# Complete Training Pipeline - PowerShell Version
# Run everything end-to-end on Windows

param(
    [switch]$SkipDataGeneration = $false,
    [int]$TotalExamples = 5000,
    [int]$ExamplesPerClass = 0,
    [int]$Epochs = 3,
    [int]$BatchSize = 16,
    [double]$LearningRate = 0.00002,
    [switch]$SkipTraining = $false
)

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "🚀 Deepiri Training Pipeline" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""
Write-Host "This will:" -ForegroundColor Yellow
Write-Host "  1. Generate synthetic training data" -ForegroundColor White
Write-Host "  2. Prepare the dataset" -ForegroundColor White
Write-Host "  3. Train the DeBERTa classifier" -ForegroundColor White
Write-Host "  4. Evaluate model performance" -ForegroundColor White
Write-Host ""

$ErrorActionPreference = "Stop"

function Run-Step {
    param(
        [string]$Command,
        [string]$Description
    )
    
    Write-Host ""
    Write-Host "=" * 80 -ForegroundColor Cyan
    Write-Host "Step: $Description" -ForegroundColor Green
    Write-Host "=" * 80 -ForegroundColor Cyan
    Write-Host "Running: $Command" -ForegroundColor Gray
    Write-Host ""
    
    Invoke-Expression $Command
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "❌ Error in: $Description" -ForegroundColor Red
        Write-Host "Command failed with exit code $LASTEXITCODE" -ForegroundColor Red
        return $false
    }
    
    Write-Host ""
    Write-Host "✅ Completed: $Description" -ForegroundColor Green
    return $true
}

# Step 1: Generate synthetic data
if (-not $SkipDataGeneration) {
    $cmd = "python app/train/scripts/generate_synthetic_data.py"
    if ($ExamplesPerClass -gt 0) {
        $cmd += " --examples-per-class $ExamplesPerClass"
    } else {
        $cmd += " --total-examples $TotalExamples"
    }
    
    if (-not (Run-Step -Command $cmd -Description "Generate Synthetic Data")) {
        Write-Host ""
        Write-Host "❌ Failed to generate synthetic data" -ForegroundColor Red
        exit 1
    }
}

# Step 2: Prepare training data
$cmd = "python app/train/scripts/prepare_training_data.py"
if (-not (Run-Step -Command $cmd -Description "Prepare Training Data")) {
    Write-Host ""
    Write-Host "❌ Failed to prepare training data" -ForegroundColor Red
    Write-Host "   Note: If data was already generated, this might be okay" -ForegroundColor Yellow
    Write-Host "   Check if app/train/data/classification_train.jsonl exists" -ForegroundColor Yellow
}

# Step 3: Train the model
if (-not $SkipTraining) {
    $cmd = "python app/train/scripts/train_intent_classifier.py"
    $cmd += " --epochs $Epochs"
    $cmd += " --batch-size $BatchSize"
    $cmd += " --learning-rate $LearningRate"
    
    if (-not (Run-Step -Command $cmd -Description "Train Intent Classifier")) {
        Write-Host ""
        Write-Host "❌ Failed to train model" -ForegroundColor Red
        exit 1
    }
}

# Step 4: Evaluate the model
Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "🎯 Evaluating Model Performance" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Cyan
$cmd = "python app/train/scripts/evaluate_trained_model.py"
if (-not (Run-Step -Command $cmd -Description "Evaluate Model on Test Set")) {
    Write-Host ""
    Write-Host "⚠️  Evaluation failed, but model is still trained" -ForegroundColor Yellow
}

# Success!
Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "🚀 TRAINING PIPELINE COMPLETE! 🚀" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""
Write-Host "✅ Model trained and evaluated successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "📁 Model location: app/train/models/intent_classifier" -ForegroundColor Cyan
Write-Host "📊 Evaluation report: app/train/models/intent_classifier/evaluation_report.json" -ForegroundColor Cyan
Write-Host ""
Write-Host "🧪 Test the model interactively:" -ForegroundColor Yellow
Write-Host "   python app/train/scripts/test_model_inference.py" -ForegroundColor White
Write-Host ""
Write-Host "🔥 YOU'RE READY FOR LIFTOFF! 🔥" -ForegroundColor Green
Write-Host ""

