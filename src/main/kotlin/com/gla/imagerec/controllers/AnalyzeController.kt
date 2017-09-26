package com.gla.imagerec.controllers

import com.gla.imagerec.recognizer.Recognized
import com.jacksierkstra.tensorflow.Recognizer
import org.springframework.web.bind.annotation.*
import org.springframework.web.multipart.MultipartFile
import org.tensorflow.Graph
import org.tensorflow.Session
import org.tensorflow.Tensor
import org.tensorflow.TensorFlow
import java.nio.charset.Charset

@RestController
class AnalyzeController {

    lateinit var recognizer: Recognizer

    init {
        try {
            recognizer = Recognizer()
            recognizer.loadGraph("tensorflow_inception_graph.pb")
            recognizer.loadLabels("imagenet_comp_graph_label_strings.txt")
        } catch (ex2: Exception) {
            println(ex2.message)
        }

    }

    @RequestMapping(value = "analyze", method = arrayOf(RequestMethod.POST))
    fun analyzeImage(@RequestParam("image") requestBody: MultipartFile): Recognized {
        return recognizer.recognize(requestBody.bytes)
    }

    @RequestMapping(value = "analyze", method = arrayOf(RequestMethod.GET))
    fun analyzeImageInstructions(): String {
        return "Please call the POST version of this url and provided an image to be analyzed."
    }


}