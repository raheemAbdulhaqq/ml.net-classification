using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace ML_Classification.DataModels
{
    class ModelInput
    {
        [LoadColumn(0)]
        public string skill1 { get; set; }
        [LoadColumn(1)]
        public string skill2 { get; set; }
        [LoadColumn(2)]
        public string skill3 { get; set; }
        [LoadColumn(3)]
        public string placementType { get; set; }
        [LoadColumn(4)]
        public string location { get; set; }
        [LoadColumn(5)]
        public string organization { get; set; }
    }
}
