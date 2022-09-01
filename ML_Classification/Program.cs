using Microsoft.ML;
using ML_Classification.DataModels;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using static Microsoft.ML.DataOperationsCatalog;

namespace ML_Classification
{
    class Program
    {
        static readonly string _dataFilePath = Path.Combine(Environment.CurrentDirectory, "Data", "trainData.csv");
        static void Main(string[] args)
        {
            //Creating MLContxet model to be shared accross model building, validation and prediction process
            MLContext mlContext = new MLContext();

            IDataView mushroomDataView = mlContext.Data.LoadFromTextFile<ModelInput>(_dataFilePath, hasHeader: true, separatorChar: ',', allowSparse: false);

            //Loading the data from csv files
            //Splitting the dataset into train/test sets
            TrainTestData mushroomTrainTestData = LoadData(mlContext, testDataFraction: 0.25);

            //Creating data transformation pipeline which transforma the data a form acceptable by model
            //Returns an object of type IEstimator<ITransformer>
            var pipeline = ProcessData(mlContext);

            //passing the transformation pipeline and training dataset to crossvalidate and build the model
            //returns the model object of type ITransformer 
            var trainedModel = BuildAndTrain(mlContext, pipeline, mushroomTrainTestData.TrainSet);

            mlContext.Model.Save(trainedModel, mushroomDataView.Schema, "Classification_Model.zip");
            //Sample datainput for predicrtion
            var Input1 = new ModelInput
            {
                skill1 = "Random1",
                skill2 = "Random2",
                skill3 = "Random3",
                placementType = "Random4",
                location = "Random5"

            };

            //Sample datainput for predicrtion
            var Input2 = new ModelInput
            {
                skill1 = "Consulting",
                skill2 = "Diet Management",
                skill3 = "Leadership",
                placementType = "",
                location = ""
            };



            //passing trained model and sample input data to make single prediction 
            var result = PredictSingleResult(mlContext, trainedModel, Input1);

            Console.WriteLine("================================= Single Prediction Result ===============================");
            // Evaluate(mlContext, pipeline, trainedModel,  mushroomTrainTestData.TestSet);
            Console.WriteLine($"Predicted Result: {result.Label}");

            Console.ReadKey();




        }

        public static TrainTestData LoadData(MLContext mlContext, double testDataFraction)
        {
            //Read data
            IDataView mushroomDataView = mlContext.Data.LoadFromTextFile<ModelInput>(_dataFilePath, hasHeader: true, separatorChar: ',', allowSparse: false);

            TrainTestData mushroomTrainTestData = mlContext.Data.TrainTestSplit(mushroomDataView, testFraction: testDataFraction);

            return mushroomTrainTestData;
        }

        public static IEstimator<ITransformer> ProcessData(MLContext mlContext)
        {
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", inputColumnName: nameof(ModelInput.organization))
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "skill1", outputColumnName: "skill1Featurized"))
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "skill2", outputColumnName: "skill2Featurized"))
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "skill3", outputColumnName: "skill3Featurized"))
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "placementType", outputColumnName: "placememntTypeFeaturized"))
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "location", outputColumnName: "locationFeaturized"))
                .Append(mlContext.Transforms.Concatenate(outputColumnName: "Features", inputColumnNames: new string[] { "skill1Featurized",
                    "skill2Featurized",
                    "skill3Featurized",
                    "placememntTypeFeaturized",
                    "locationFeaturized" }));
            return pipeline;
        }

        public static ITransformer BuildAndTrain(MLContext mlContext, IEstimator<ITransformer> pipeline, IDataView trainDataView)
        {

            //   PeekDataViewInConsole(mlContext, trainDataView, pipeline, 2);




            var trainPipeline = pipeline.Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.AveragedPerceptron("Label", "Features", numberOfIterations: 10)))
                                        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            Console.WriteLine("=============== Starting 10 fold cross validation ===============");
            var crossValResults = mlContext.MulticlassClassification.CrossValidate(data: trainDataView, estimator: trainPipeline, numberOfFolds: 10, labelColumnName: "Label");

            var metricsInMultipleFolds = crossValResults.Select(r => r.Metrics);

            var microAccuracyValues = metricsInMultipleFolds.Select(m => m.MicroAccuracy);
            var microAccuracyAverage = microAccuracyValues.Average();


            var macroAccuracyValues = metricsInMultipleFolds.Select(m => m.MacroAccuracy);
            var macroAccuracyAverage = macroAccuracyValues.Average();


            var logLossValues = metricsInMultipleFolds.Select(m => m.LogLoss);
            var logLossAverage = logLossValues.Average();


            var logLossReductionValues = metricsInMultipleFolds.Select(m => m.LogLossReduction);
            var logLossReductionAverage = logLossReductionValues.Average();


            //Console.WriteLine($"*************************************************************************************************************");
            //Console.WriteLine($"*       Metrics Classification model      ");
            //Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            //Console.WriteLine($"*       Average MicroAccuracy:    {microAccuracyAverage:0.###} ");
            //Console.WriteLine($"*       Average MacroAccuracy:    {macroAccuracyAverage:0.###} ");
            //Console.WriteLine($"*       Average LogLoss:          {logLossAverage:#.###} ");
            //Console.WriteLine($"*       Average LogLossReduction: {logLossReductionAverage:#.###} ");
            //Console.WriteLine($"*************************************************************************************************************");


            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = trainPipeline.Fit(trainDataView);
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();

            return model;

        }

        public static ModelPrediction PredictSingleResult(MLContext mlContext, ITransformer model, ModelInput input)
        {

        //    ITransformer loadedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);

        //    GitHubIssue singleIssue = new GitHubIssue() { Title = "Entity Framework crashes", Description = "When connecting to the database, EF is crashing" }; // Our single issue
        //    _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(loadedModel);
        //    var singleprediction = _predEngine.Predict(singleIssue);
        //    Console.WriteLine($"=============== Single Prediction - Result: {singleprediction.Area} ===============");

            //Creating the prediction engine which takes data model input and output
            var predictEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelPrediction>(model);

            var predOutput = predictEngine.Predict(input);

            return predOutput;


        }

        ////public static ModelPrediction PredictMultipleResult(MLContext mlContext, ITransformer model, ModelInput inputs)
        //{

        //    ////Creating the prediction engine which takes data model input and output
        //    //var predictEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelPrediction>(model);

        //    //var predOutput = predictEngine.Predict(input);

        //    //return predOutput;

        //    IDataView batchIssues = mlContext.Data.LoadFromEnumerable(inputs);


        //    IEnumerable<ModelInput> predictedResults = mlContext.Data.CreateEnumerable<ModelInput>(batchIssues, reuseRowObject: false);

        //    foreach (ModelInput prediction in predictedResults)
        //    {
        //        Console.WriteLine(prediction.organization);
        //    }

        //}


        // This method using 'DebuggerExtensions.Preview()' should only be used when debugging/developing, not for release/production trainings

        //     .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));
    }
}
