package com.jacksierkstra.tensorflow

import com.gla.imagerec.recognizer.Recognized


interface Recognizable {

    @Throws(Exception::class)
    fun recognize(file: ByteArray) : Recognized

    @Throws(Exception::class)
    fun recognize(file: String)

    @Throws(Exception::class)
    fun loadGraph(file: String)

    @Throws(Exception::class)
    fun loadLabels(file: String)
}
