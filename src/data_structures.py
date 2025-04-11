# Neural network architectures database
model_architectures = {
    "ResNet-50": {
        "type": "CNN",
        "layers": [
            {"type": "Conv2D", "filters": 64, "kernel_size": 7, "stride": 2},
            {"type": "BatchNormalization"},
            {"type": "ReLU"},
        ],
        "use_cases": ["image classification", "computer vision"],
        "paper_reference": "Deep Residual Learning for Image Recognition",
        "performance_metrics": {"ImageNet": {"top1": 75.3, "top5": 92.2}},
    },
}

# Debugging knowledge database
debugging_database = [
    {
        "problem": "Vanishing gradients",
        "symptoms": ["Loss plateaus early", "Deeper layers show minimal updates"],
        "causes": ["Deep networks with sigmoid/tanh activations", "Poor initialization"],
        "solutions": [
            {
                "solution": "Use ReLU or leaky ReLU activation functions",
                "explanation": "These activations have non-zero gradients for positive inputs",
                "code_example": "model.add(tf.keras.layers.Dense(128, activation='relu'))",
            },
        ],
    },
]
