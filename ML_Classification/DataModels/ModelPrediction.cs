using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace ML_Classification.DataModels
{
    class ModelPrediction
    {
        [ColumnName("PredictedLabel")]
        public string Label { get; set; }
        public float[] Score { get; set; }
    }
}
