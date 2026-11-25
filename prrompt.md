Refactor process_diffusion() into a RESTful API endpoint.

Endpoint:
GET /sample

Query Parameters:
- mask_text (string, required)
- N (integer, optional, default=10)

Behavior:
- Call process_diffusion(mask_text, N)
- Return generated samples as JSON

Response Format:

{
  "data": [
    "sample1",
    "sample2",
    "sample3"
  ]
}

Requirements:
- Follow RESTful API conventions.
- Validate N >= 1.
- Return HTTP 400 for invalid parameters.
- Return application/json.
- Keep process_diffusion() as the business logic layer.
- Move request parsing and response formatting into the API layer.