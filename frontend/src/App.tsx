import MainPage from './atomic/pages/MainPage/MainPage.tsx'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import LectionPage from './atomic/pages/LectionPage/LectionPage.tsx'
import ResultsPage from './atomic/pages/ResultsPage/ResultsPage.tsx'

function App() {
    return (
        <BrowserRouter>
            <Routes>
                <Route path="/" element={<MainPage />} />
                <Route path="/lection" element={<LectionPage />} />
                <Route path="/results" element={<ResultsPage />} />
            </Routes>
        </BrowserRouter>
    )
}

export default App
