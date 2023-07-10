import React, { useRef, useEffect } from 'react'

const InputBar = ({ input, setInput, handleKeyDown, handleSubmit }) => {
  const inputRef = useRef(null)

  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto'
      inputRef.current.style.height = inputRef.current.scrollHeight + 'px'
    }
  }, [input])

  return (
    <div>
      <form onSubmit={handleSubmit} className="flex items-center px-4 py-2 justify-center md:px-4 md:py-4">
        <div className="w-full md:w-1/2 max-w-xl flex items-center">
          <textarea
            ref={inputRef}
            rows="1"
            placeholder="What is developer marketing?"
            className="flex-1 p-2 border rounded-lg focus:outline-none focus:ring focus:border-blue-300 resize-none overflow-hidden bg-gray-600 text-gray-100"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
          />
          <button
            type="submit"
            className="ml-2 px-2 py-1 rounded-lg bg-blue-500 text-white focus:outline-none hover:bg-blue-600 md:ml-4 md:px-4 md:py-2"
          >
            Send
          </button>
        </div>
      </form>
      <div className="pb-2 text-center text-xs text-gray-400 md:pb-4">
        The app currently references the HTML provided a lot right now, because the prompt and data hygiene is not dialed in.
      </div>
    </div>
  )
}

export default InputBar