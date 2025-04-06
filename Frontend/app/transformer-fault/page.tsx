'use client'
import { useState } from "react";
import axios from "axios";
import Link from 'next/link';

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL;

export default function TransformerFault() {
  const [trainFile, setTrainFile] = useState<File | null>(null);
  const [predictFile, setPredictFile] = useState<File | null>(null);
  const [trainMessage, setTrainMessage] = useState("");
  const [predictMessage, setPredictMessage] = useState("");
  const [predictions, setPredictions] = useState([]);
  //const [rawResponse, setRawResponse] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);

  const handleTrainFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      setTrainFile(event.target.files[0]);
    }
  };
  
  const handlePredictFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      setPredictFile(event.target.files[0]);
    }
  };

  const handleTrainLocalFaultModel = async () => {
    if (!trainFile) {
      setTrainMessage("Please select a training file first.");
      return;
    }

    setIsTraining(true);
    setProgress(10);

    const formData = new FormData();
    formData.append("file", trainFile);

    try {
      const response = await axios.post(`${BACKEND_URL}/train_local_fault_model`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            setProgress(percentCompleted);
          }
        },
      });

      setProgress(100);
      setTrainMessage(`Training Complete! \n Accuracy: ${response.data["Accuracy"]},\n F1 Score: ${response.data["F1 Score"]}`);
    } catch (error) {
      setTrainMessage("Training failed. Please try again.");
      console.error(error);
    } finally {
      setTimeout(() => {
        setIsTraining(false);
        setProgress(0);
      }, 1500);
    }
  };

  const handleGetFaultPredictions = async () => {
    if (!predictFile) {
      setPredictMessage("Please select a prediction file first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", predictFile);

    try {
      const response = await axios.post(`${BACKEND_URL}/predict_fault`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      
      // Log the full response for debugging
      // console.log("Full Response:", response.data);
      // setRawResponse(response.data);

      // Extract predictions from the specific key
      const fetchedPredictions = response.data?.fault_predictions || [];
      setPredictions(fetchedPredictions);
      
      setPredictMessage(
        fetchedPredictions.length > 0 
          ? "Fault Predictions fetched successfully." 
          : "No predictions returned. Check your dataset."
      );
    } catch (error) {
      setPredictMessage("Fault Prediction failed. Please try again.");
      setPredictions([]); 
      console.error("Error details:", error);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-5">
      <div className="bg-white p-6 rounded-lg shadow-md w-full max-w-md">
        <div className="flex justify-between items-center mb-4">
          <h1 className="text-2xl font-bold">Transformer Fault Detection</h1>
          <Link href="/" className="text-green-500 hover:underline">
            Home
          </Link>
        </div>
        
        <h2 className="text-lg font-semibold">Upload Training Dataset:</h2>
        <input type="file" onChange={handleTrainFileChange} className="mb-4 w-full border p-2 rounded" />

        <button 
          onClick={handleTrainLocalFaultModel} 
          className="w-full bg-green-500 text-white p-2 rounded mt-2 hover:bg-green-600 hover:cursor-pointer"
          disabled={isTraining}
        >
          {isTraining ? "Training..." : "Train Fault Detection Model"}
        </button>

        {isTraining && (
          <div className="w-full bg-gray-200 rounded-full h-2.5 mt-3">
            <div 
              className="bg-green-600 h-2.5 rounded-full transition-all duration-300 ease-in-out"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
        )}

        {trainMessage && (
          <div className="mt-4 bg-gray-200 p-4 rounded">
            <h3 className="text-lg font-semibold">Training Status:</h3>
            <p className="text-gray-800">{trainMessage}</p>
          </div>
        )}

        <h2 className="text-lg font-semibold mt-6">Upload Prediction Dataset:</h2>
        <input type="file" onChange={handlePredictFileChange} className="mb-4 w-full border p-2 rounded" />

        <button 
          onClick={handleGetFaultPredictions} 
          className="w-full bg-red-500 text-white p-2 rounded mt-2 hover:bg-red-600 hover:cursor-pointer"
          disabled={isTraining}
        >
          Get Fault Predictions
        </button>

        {predictMessage && (
          <div className="mt-4 bg-gray-200 p-4 rounded">
            <h3 className="text-lg font-semibold">Prediction Status:</h3>
            <p className="text-gray-800">{predictMessage}</p>
          </div>
        )}

        {predictions.length > 0 && (
          <div className="mt-4 bg-gray-200 p-4 rounded">
            <h3 className="text-lg font-semibold">Fault Predictions:</h3>
            <ul className="mt-2">
              {predictions.map((pred, index) => (
                <li key={index} className="text-gray-800">
                  Sample {index + 1}: {pred === 1 ? "Fault Detected" : "No Fault"}
                </li>
              ))}
            </ul>
            <div className="mt-2 text-sm text-gray-600">
              Total Faults: {predictions.filter(p => p === 1).length} 
              / {predictions.length}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}