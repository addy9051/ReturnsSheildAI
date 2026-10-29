@echo off
rem ReturnShield AI - TensorRT Engine Compilation Script
echo Building TensorRT FP8 Engine from dinov2_vits14_fp8.onnx...
trtexec --onnx=D:\Downloads\projects\ReturnsSheildAI\storage\models\dinov2_vits14_fp8.onnx --saveEngine=D:\Downloads\projects\ReturnsSheildAI\storage\models\dinov2_vits14_fp8.engine --minShapes=input:1x3x224x224 --optShapes=input:4x3x224x224 --maxShapes=input:16x3x224x224 --fp8 --fp16
echo TensorRT build complete.
