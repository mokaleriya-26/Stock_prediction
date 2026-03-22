# core/views.py

import os
import json
import requests
from django.shortcuts import render , redirect , HttpResponse
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from predictor.views import COMPANY_NAMES
from django.http import JsonResponse
from datetime import datetime, timedelta
from predictor.views import COMPANY_NAMES, fetch_company_news
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt

def home(request):
    """View for the Main Page (Home)"""
    return render(request, 'index.html')
def signup_view(request):
    print("Signup view triggered with method:", request.method)
    if request.method == 'POST':
        print("POST data:", request.POST)
        name = request.POST.get('name')
        email = request.POST.get('email')
        password = request.POST.get('password')
        print(f"Data received: {name}, {email}, {password}")

        if User.objects.filter(username=email).exists():
            print("User already exists!")
            messages.error(request, 'Email already registered!')
            return redirect('signup')

        user = User.objects.create_user(
            username=email,
            email=email,
            password=password,
            first_name=name
        )
        user.save()
        print("User created successfully!")
        messages.success(request, 'Account created successfully! Please sign in.')
        print("✅ Redirecting to signin...")
        return redirect('signin')

    return render(request, 'signup.html')

def signin_view(request):
   if request.method == 'POST':
        print("Signin POST triggered")
        email = request.POST.get('email')
        password = request.POST.get('password')
        print(f"Email: {email}, Password: {password}")

        user = authenticate(username=email, password=password)
        if user is not None:
            login(request, user)
            messages.success(request, f'Welcome, {user.first_name}!')
            return redirect('home')
        else:
            messages.error(request, 'Invalid email or password.')
            return redirect('signin')

   return render(request, 'signin.html')
def signout_view(request):
    """Handle user sign out"""
    logout(request)
    messages.info(request, 'You have been signed out successfully!')
    return redirect('signin')
def about(request): 
    """View for the About Page"""
    return render(request, 'about.html')
def market_data(request):
    """View for the Market Data Page"""
    return render(request, 'market_data.html')
def privacy_policy(request):
    """View for the Privacy Policy Page"""
    return render(request, 'privacy_policy.html',{})
def terms_of_service(request):
    """View for the Terms of Service Page"""    
    return render(request, 'terms_of_service.html',{})
@login_required(login_url='/signin/')
def analysis(request):
    # return HttpResponse("Analysis Page executed")
    return render(request, 'analysis.html', {})
def comparison(request):
    return render(request, "comparison.html", {
        "companies": COMPANY_NAMES
    })
def sector_sentiment_api(request):
    """
    Returns average sentiment sector-wise for homepage chips.
    """
    sector_map = {
        "Banking": ["HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS", "KOTAKBANK.NS", "SBIN.NS"],
        "IT": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"],
        "Energy": ["RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "COALINDIA.NS"],
        "Auto": ["MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "TMPV.NS"],
        "Pharma": ["SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "APOLLOHOSP.NS"],
        "FMCG": ["ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "TATACONSUM.NS"],
        "Infrastructure": ["LT.NS", "ADANIENT.NS", "ADANIPORTS.NS", "GRASIM.NS", "ULTRACEMCO.NS"],
    }

    news_end = datetime.now()
    news_start = news_end - timedelta(days=30)

    result = []

    for sector, tickers in sector_map.items():
        scores = []

        for ticker in tickers:
            try:
                sentiment_score, headlines = fetch_company_news(
                    ticker,
                    news_start.strftime('%Y-%m-%d'),
                    news_end.strftime('%Y-%m-%d'),
                    "en"
                )
                scores.append(sentiment_score)
            except Exception as e:
                print(f"Sector sentiment error for {ticker}: {e}")

        avg_score = sum(scores) / len(scores) if scores else 0.0

        if avg_score > 0.15:
            label = "Bullish"
        elif avg_score < -0.15:
            label = "Bearish"
        else:
            label = "Neutral"

        result.append({
            "sector": sector,
            "score": round(avg_score, 3),
            "label": label
        })

    result = sorted(result, key=lambda x: x["score"], reverse=True)
    print(f"\n--- Sector: {sector} ---")

    return JsonResponse({"sectors": result})
@require_POST
@login_required(login_url='/signin/')
def generate_verdict_explanation(request):
    try:
        print("=== generate_verdict_explanation called ===")

        body = json.loads(request.body)
        winner = body.get("winner", {})
        stocks = body.get("stocks", [])

        print("Winner received:", winner)
        print("Stocks received:", len(stocks))

        if not winner or not stocks:
            return JsonResponse({
                "success": False,
                "error": "Missing winner or stocks data."
            }, status=400)

        api_key = os.getenv("GEMINI_API_KEY")
        print("GEMINI_API_KEY loaded:", bool(api_key))

        if not api_key:
            return JsonResponse({
                "success": False,
                "error": "Gemini API key not found."
            }, status=500)

        prompt = f"""
You are a stock comparison explanation assistant.

Winner stock:
{json.dumps(winner, indent=2)}

All compared stocks:
{json.dumps(stocks, indent=2)}

Write a long, detailed, and easy-to-understand explanation of the comparison result.

Requirements:
- Clearly state which stock ranked first and mention the score difference versus the other selected stocks.
- Explain in detail why the winner ranked higher using the actual values of P/E, Forward P/E, Beta, Market Cap, and Volume.
- Compare the winner with each of the other stocks and explain which specific metrics were stronger or weaker.
- Explain what the result means in simple language for someone reviewing the stock comparison.
- Keep the explanation natural, direct, and informative.
- Write 4 to 6 full paragraphs.
- Each paragraph should have 3 to 5 sentences.
- Use actual numbers from the input data.
- Do not invent any values.
- Do not use bullet points.
- Do not use headings.
- Output clean HTML using only <p> and <strong> tags.
"""

        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key={api_key}",
            headers={
                "Content-Type": "application/json",
            },
            json={
                "contents": [
                    {
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.5,
                    "topP": 0.9,
                    "maxOutputTokens": 20000
                }
            },
            timeout=30
        )

        print("Gemini status code:", response.status_code)

        if response.status_code != 200:
            fallback_explanation = (
                f"<p><strong>{winner.get('ticker', 'This stock')}</strong> ranked highest "
                f"based on the internal scoring model using valuation, risk, liquidity, and company size metrics.</p>"
                f"<ul>"
                f"<li>P/E: {winner.get('pe', '-')}</li>"
                f"<li>Forward P/E: {winner.get('forwardPe', '-')}</li>"
                f"<li>Beta: {winner.get('beta', '-')}</li>"
                f"<li>Market Cap: {winner.get('marketCap', '-')}</li>"
                f"<li>Volume: {winner.get('volume', '-')}</li>"
                f"</ul>"
                f"<p>The AI explanation service is currently unavailable, so this summary is generated from available stock metrics. Please do your own research before investing.</p>"
            )

            return JsonResponse({
                "success": True,
                "explanation": fallback_explanation,
                "source": "fallback",
                "llm_error": response.text
            })

        result = response.json()
        explanation = result["candidates"][0]["content"]["parts"][0]["text"]

        return JsonResponse({
            "success": True,
            "explanation": explanation,
            "source": "gemini"
        })

    except Exception as e:
        print("ERROR in generate_verdict_explanation:", str(e))
        return JsonResponse({
            "success": False,
            "error": str(e)
        }, status=500)