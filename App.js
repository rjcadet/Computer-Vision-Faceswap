import React, { useState, useEffect } from "react";
import { ScrollView, View, Button, Image, StyleSheet, Text, ActivityIndicator, Alert } from "react-native";
import * as ImagePicker from "expo-image-picker";

export default function App() {
  const [image1, setImage1] = useState(null);
  const [image2, setImage2] = useState(null);
  const [swappedImage, setSwappedImage] = useState(null);
  const [loading, setLoading] = useState(false);

  // library permissions
  useEffect(() => {
    (async () => {
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      console.log("Media library permission status:", status);
      if (status !== "granted") {
        Alert.alert(
          "Permission needed",
          "Sorry, we need camera roll permissions to make this work!"
        );
      }
    })();
  }, []);

  // Pick image +debug
  const pickImage = async (setter) => {
  try {
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: false,
      quality: 1,
    });

    console.log("Picker result:", result);

    if (result.canceled) {
      console.log("Image picking canceled by user");
      return;
    }

    setter(result.assets[0].uri);
  } catch (error) {
    console.log("Error picking image:", error);
    Alert.alert("Error", "Error picking image. Check console for details.");
  }
};


  // backend
  const swapFaces = async () => {
    if (!image1 || !image2) {
      Alert.alert("Select images", "Please select both images first!");
      return;
    }

    setLoading(true);
    try {
      const formData = new FormData();

      formData.append("image1", {
        uri: image1,
        name: "image1.jpg",
        type: "image/jpeg",
      });

      formData.append("image2", {
        uri: image2,
        name: "image2.jpg",
        type: "image/jpeg",
      });

      const backendUrl = "http://!!!!!REPLACE IP HERE!!!!!:8000/swap";

      const response = await fetch(backendUrl, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Face swap failed");
      }

      const blob = await response.blob();
      const reader = new FileReader();
      reader.onloadend = () => setSwappedImage(reader.result);
      reader.readAsDataURL(blob);

    } catch (error) {
      console.log(error);
      Alert.alert(
        "Swap failed",
        "Error swapping faces. Make sure your backend is running and images have clear faces."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Face Swapper</Text>

      <Button title="Pick Image 1" onPress={() => pickImage(setImage1)} />
      {image1 && <Image source={{ uri: image1 }} style={styles.image} />}

      <Button title="Pick Image 2" onPress={() => pickImage(setImage2)} />
      {image2 && <Image source={{ uri: image2 }} style={styles.image} />}

      <Button title="Swap Faces" onPress={swapFaces} />

      {loading && <ActivityIndicator size="large" color="#0000ff" />}

      {swappedImage && (
        <>
          <Text style={{ marginTop: 20 }}>Swapped Image:</Text>
          <Image source={{ uri: swappedImage }} style={styles.image} />
        </>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    paddingTop: 50,
    paddingHorizontal: 20,
    alignItems: "center",
    backgroundColor: "#fff",
  },
  title: {
    fontSize: 24,
    fontWeight: "bold",
    marginBottom: 20,
  },
  image: {
    width: 250,
    height: 250,
    resizeMode: "contain",
    marginVertical: 10,
  },
});
